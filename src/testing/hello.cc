#include <tvm/ffi/function.h>
#include <tvm/ir/expr.h>

#include <string>



namespace tcl {

using namespace tvm;

std::string HelloWorld() {
  return "Hello, World!";
}

TVM_FFI_REGISTER_GLOBAL("testing.HelloWorld")
    .set_body_typed([]{
        return HelloWorld();
    });



} // namespace tcl








