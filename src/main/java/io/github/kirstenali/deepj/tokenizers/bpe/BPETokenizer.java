package io.github.kirstenali.deepj.tokenizers.bpe;

import io.github.kirstenali.deepj.tokenizers.Tokenizer;

import java.io.ByteArrayOutputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

public record BPETokenizer(BPEModel model) implements Tokenizer {

    @Override
    public int[] encode(String text) {
        List<Integer> ids = new ArrayList<>();

        for (String piece : BPEBytes.splitPreserveWhitespace(text)) {
            ids.addAll(encodePiece(piece));
        }

        return ids.stream().mapToInt(Integer::intValue).toArray();
    }

    @Override
    public String decode(int[] ids) {
        ByteArrayOutputStream out = new ByteArrayOutputStream();

        for (int id : ids) {
            if (id < 0 || id >= model.idToBytes().size()) {
                throw new IllegalArgumentException("Token id out of range: " + id);
            }

            if (id == model.endOfWordId()) {
                continue;
            }

            byte[] bytes = model.idToBytes().get(id);
            out.write(bytes, 0, bytes.length);
        }

        return out.toString(StandardCharsets.UTF_8);
    }

    @Override
    public int vocabSize() {
        return model.vocabSize();
    }

    private List<Integer> encodePiece(String piece) {
        List<Integer> tokens = BPETrainer.toTokenIds(piece, model.endOfWordId());

        for (TokenPair merge : model.merges()) {
            int mergedId = model.mergeToNewId().get(merge);
            tokens = applyMerge(tokens, merge, mergedId);
        }

        if (!tokens.isEmpty() && tokens.get(tokens.size() - 1) == model.endOfWordId()) {
            tokens.remove(tokens.size() - 1);
        }

        return tokens;
    }

    private List<Integer> applyMerge(List<Integer> input, TokenPair pair, int newId) {
        List<Integer> out = new ArrayList<>(input.size());
        int i = 0;

        while (i < input.size()) {
            if (i < input.size() - 1
                    && input.get(i) == pair.left()
                    && input.get(i + 1) == pair.right()) {
                out.add(newId);
                i += 2;
            } else {
                out.add(input.get(i));
                i++;
            }
        }

        return out;
    }
}