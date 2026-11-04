package io.github.kirstenali.deepj.nn.transformer;

/** The user prompt and visible reply of a completed chat turn. */
public interface ConversationTurn {
    String prompt();
    String reply();
}
