#include "ggml-impl.h"

#include <cstdlib>
#include <exception>

static std::terminate_handler previous_terminate_handler;

LM_GGML_NORETURN static void lm_ggml_uncaught_exception() {
    lm_ggml_print_backtrace();
    if (previous_terminate_handler) {
        previous_terminate_handler();
    }
    abort(); // unreachable unless previous_terminate_handler was nullptr
}

static bool lm_ggml_uncaught_exception_init = []{
    const char * LM_GGML_NO_BACKTRACE = getenv("LM_GGML_NO_BACKTRACE");
    if (LM_GGML_NO_BACKTRACE) {
        return false;
    }
    const auto prev{std::get_terminate()};
    LM_GGML_ASSERT(prev != lm_ggml_uncaught_exception);
    previous_terminate_handler = prev;
    std::set_terminate(lm_ggml_uncaught_exception);
    return true;
}();
