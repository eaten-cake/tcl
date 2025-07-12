import inspect
import ast
import astunparse

from typing import Dict, Any, List

import tempfile

import tvm
from tvm import relax
from tvm import dlight as dl

from tvm.script.ir_builder import (
    IRBuilder,
    ir as I,
    relax as relax_builder,
    tir as T,
)


class CodeGenerator(ast.NodeVisitor):

    def __init__(self, fn_ast: ast.Module, ctx, target: str = "cpu"):
        self.fn_ast = fn_ast
        self.ctx = ctx
        self.target = target
        self.ib = IRBuilder()
        self.ir_module = None
        self.entry = None
        self.ret = None
        self.local_var_table: Dict[str, Any] = {}

    def code_gen(self):
        with self.ib:
            self.visit(self.fn_ast)
        module: tvm.IRModule = self.ib.get()
        seq = tvm.transform.Sequential(
            [
                relax.transform.ConvertToDataflow(),
                relax.transform.LegalizeOps(),
                relax.transform.AnnotateTIROpPattern(),
                relax.transform.FuseOps(),
                relax.transform.FuseTIR(),
            ]
        )
        module = seq(module)
        mapped_target = {"cpu": "llvm", "cuda": "nvidia/geforce-rtx-3060"}
        target = tvm.target.Target(mapped_target[self.target])
        if "cuda" in target.keys:
            with target:
                module = dl.ApplyDefaultSchedule(
                    dl.gpu.Fallback(),
                )(module)
                # work_dir = tempfile.TemporaryDirectory().name
                # module = relax.transform.MetaScheduleTuneIRMod(
                #     params={},
                #     work_dir=work_dir,
                #     max_trials_global=10,
                # )(module)
                # module = relax.transform.MetaScheduleApplyDatabase(work_dir=work_dir)(
                #     module
                # )

        module.show()

        ex = relax.build(module, target)

        device = tvm.cuda(0) if "cuda" in target.keys else tvm.cpu(0)
        vm = relax.VirtualMachine(ex, device)
        return vm[self.entry]

    def visit(self, node):
        # print("Visit " + node.__class__.__name__)
        return super().visit(node)

    def visit_Module(self, node: ast.Module):
        assert self.ir_module is None, "IR module already created"
        self.ir_module = I.ir_module()
        with self.ir_module:
            super().generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        print(ast.dump(node, indent=4))
        fn = relax_builder.function()
        self.entry = node.name
        with fn:
            relax_builder.func_name(node.name)
            self.visit(node.args)
            self._visit_compound_stmt(node.body)

            if self.ret is None:
                relax_builder.func_ret_value(relax.ShapeExpr([]))
            else:
                relax_builder.func_ret_value(self.ret)

    def visit_Pass(self, node: ast.Pass):
        pass

    def _visit_compound_stmt(self, stmts):
        assert isinstance(
            stmts, (list, tuple)
        ), "Expected a list or tuple of statements"
        for stmt in stmts:
            ret = self.visit(stmt)
            if ret is not None and isinstance(stmt, ast.Return):
                self.ret = ret

    def visit_Assign(self, node: ast.Assign):
        assert len(node.targets) == 1, "Only single assignment is supported"
        target: relax.Var = self.visit(node.targets[0])
        value = self.visit(node.value)
        self.local_var_table[target.name_hint] = value
        self.ib.name(target.name_hint, value)

    def visit_Name(self, node: ast.Name) -> relax.Var:
        name = node.id
        if isinstance(node.ctx, ast.Store):
            if name not in self.local_var_table:
                self.local_var_table[name] = relax.Var(
                    name, struct_info=relax.ObjectStructInfo()
                )
        return self.local_var_table[name]

    def visit_BinOp(self, node: ast.BinOp):
        lhs = self.visit(node.left)
        rhs = self.visit(node.right)
        op = node.op
        return relax_builder.emit(self._binOp_maker(op)(lhs, rhs))

    def _binOp_maker(self, node: ast.operator):
        if isinstance(node, ast.Add):
            return relax_builder.add
        elif isinstance(node, ast.Mult):
            return relax_builder.multiply
        else:
            raise NotImplementedError(
                f"Operator {node.__class__.__name__} not implemented"
            )

    def visit_Constant(self, node: ast.Constant) -> relax.Var:
        value = node.value
        relax_const = None
        if isinstance(value, (int, float)):
            relax_const = relax.const(value)
        elif isinstance(value, str):
            # 字符串类型当前仅用作print函数，故直接返回，而不是emit
            relax_const = relax.StringImm(value)
            return relax_const
        else:
            raise NotImplementedError(
                f"Constant type {type(value)} not implemented"
            )
        return relax_builder.emit(relax_const)

    def visit_Return(self, node: ast.Return):
        return self.visit(node.value)

    def visit_arguments(self, node: ast.arguments):
        for arg in node.args:
            assert arg.annotation is not None, "Argument annotation is required"
            arg_name = arg.arg
            anno = eval(astunparse.unparse(arg.annotation), self.ctx)
            param = relax_builder.arg(
                arg_name, relax.TensorStructInfo(anno.shape, anno.dtype)
            )
            self.local_var_table[arg_name] = param

    def visit_Expr(self, node: ast.Expr):
        expr = self.visit(node.value)
        return relax_builder.emit(expr)

    def visit_Call(self, node: ast.Call):
        relax_func = None
        if node.func.id == "print":
            args = node.args
            values: List[relax.Expr] = list([])
            for arg in args:
                values.append(self.visit(arg))
            relax_func = relax_builder.emit(relax.op.print(*values))
        return relax_func

    # def visit_If(self, node: ast.If):
    #     condition = self.visit(node.test)
        
    #     exit(0)
    
    # def visit_Compare(self, node: ast.Compare):
    #     print(ast.dump(node, indent=4))


class JIT:

    def __init__(self, fn, target: str = "cpu"):
        self.fn = fn
        self.target = target

    def __call__(self, *args, **kwargs):
        fn_src = inspect.getsource(self.fn)
        fn_ast = ast.parse(fn_src)
        # print(ast.dump(fn_ast, indent=4))
        ctx = self.fn.__globals__.copy()
        code_generator = CodeGenerator(fn_ast, ctx, target=self.target)
        input_args = []
        for arg in args:
            input_args.append(arg.data)
        compiled_kernel = code_generator.code_gen()
        return compiled_kernel(*input_args)


def jit(target: str = "cpu"):
    assert target in ["cpu", "cuda"], "Target must be 'cpu' or 'cuda'"

    def inner(fn):
        return JIT(fn, target=target)

    return inner
