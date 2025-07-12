#include <tvm/ffi/function.h>

#include <string>

std::string HelloWorld() {
  return "Hello, World!";
}

TVM_FFI_REGISTER_GLOBAL("testing.HelloWorld")
    .set_body_typed([]{
        return HelloWorld();
    });







