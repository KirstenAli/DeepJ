package io.github.kirstenali.deepj.chatui;

/** Consistent error descriptions for text and image generation. */
final class DesktopErrors {
    private DesktopErrors() {}

    static String message(Throwable error) {
        return error == null || error.getMessage() == null ? "Unknown error" : error.getMessage();
    }
}
