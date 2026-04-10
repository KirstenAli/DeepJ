package io.github.kirstenali.deepj.tokenizers.bpe;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public final class BPEModelIO {

    private static final int MAGIC = 0x444A4250; // DJBP

    private BPEModelIO() {
    }

    public static void save(Path path, BPEModel model) throws IOException {
        try (DataOutputStream out = new DataOutputStream(new BufferedOutputStream(Files.newOutputStream(path)))) {
            out.writeInt(MAGIC);
            out.writeInt(model.modelFormatVersion());
            out.writeInt(model.endOfWordId());

            writeVocab(out, model.idToBytes());
            writeTokenKeyToId(out, model.tokenKeyToId());
            writeMerges(out, model.merges());
            writeMergeToNewId(out, model.mergeToNewId());
            writeSpecialTokens(out, model.specialTokenToId());
        }
    }

    public static BPEModel load(Path path) throws IOException {
        try (DataInputStream in = new DataInputStream(new BufferedInputStream(Files.newInputStream(path)))) {
            int magic = in.readInt();
            if (magic != MAGIC) {
                throw new IOException("Invalid BPE model file magic");
            }

            int version = in.readInt();
            if (version != BPEModel.CURRENT_FORMAT_VERSION) {
                throw new IOException("Unsupported BPE model version: " + version);
            }

            int endOfWordId = in.readInt();
            List<byte[]> idToBytes = readVocab(in);
            Map<String, Integer> tokenKeyToId = readTokenKeyToId(in);
            List<TokenPair> merges = readMerges(in);
            Map<TokenPair, Integer> mergeToNewId = readMergeToNewId(in);
            Map<String, Integer> specialTokenToId = readSpecialTokens(in);

            return new BPEModel(
                    idToBytes,
                    tokenKeyToId,
                    merges,
                    mergeToNewId,
                    endOfWordId,
                    version,
                    specialTokenToId
            );
        }
    }

    private static void writeVocab(DataOutputStream out, List<byte[]> vocab) throws IOException {
        out.writeInt(vocab.size());
        for (byte[] token : vocab) {
            out.writeInt(token.length);
            out.write(token);
        }
    }

    private static List<byte[]> readVocab(DataInputStream in) throws IOException {
        int size = in.readInt();
        List<byte[]> vocab = new ArrayList<>(size);
        for (int i = 0; i < size; i++) {
            int len = in.readInt();
            requireNonNegativeLength(len, "vocab");
            byte[] token = readExactBytes(in, len, "vocab bytes");
            vocab.add(token);
        }
        return vocab;
    }

    private static void writeTokenKeyToId(DataOutputStream out, Map<String, Integer> tokenKeyToId) throws IOException {
        writeStringIntMap(out, tokenKeyToId);
    }

    private static Map<String, Integer> readTokenKeyToId(DataInputStream in) throws IOException {
        return readStringIntMap(in);
    }

    private static void writeMerges(DataOutputStream out, List<TokenPair> merges) throws IOException {
        out.writeInt(merges.size());
        for (TokenPair pair : merges) {
            out.writeInt(pair.left());
            out.writeInt(pair.right());
        }
    }

    private static List<TokenPair> readMerges(DataInputStream in) throws IOException {
        int size = in.readInt();
        List<TokenPair> merges = new ArrayList<>(size);
        for (int i = 0; i < size; i++) {
            merges.add(new TokenPair(in.readInt(), in.readInt()));
        }
        return merges;
    }

    private static void writeMergeToNewId(DataOutputStream out, Map<TokenPair, Integer> mergeToNewId) throws IOException {
        out.writeInt(mergeToNewId.size());
        List<Map.Entry<TokenPair, Integer>> entries = new ArrayList<>(mergeToNewId.entrySet());
        entries.sort(Map.Entry.comparingByKey());
        for (Map.Entry<TokenPair, Integer> e : entries) {
            out.writeInt(e.getKey().left());
            out.writeInt(e.getKey().right());
            out.writeInt(e.getValue());
        }
    }

    private static Map<TokenPair, Integer> readMergeToNewId(DataInputStream in) throws IOException {
        int size = in.readInt();
        Map<TokenPair, Integer> map = new LinkedHashMap<>(size);
        for (int i = 0; i < size; i++) {
            TokenPair pair = new TokenPair(in.readInt(), in.readInt());
            map.put(pair, in.readInt());
        }
        return map;
    }

    private static void writeSpecialTokens(DataOutputStream out, Map<String, Integer> specialTokenToId) throws IOException {
        writeStringIntMap(out, specialTokenToId);
    }

    private static Map<String, Integer> readSpecialTokens(DataInputStream in) throws IOException {
        return readStringIntMap(in);
    }

    private static void writeStringIntMap(DataOutputStream out, Map<String, Integer> map) throws IOException {
        out.writeInt(map.size());
        List<Map.Entry<String, Integer>> entries = new ArrayList<>(map.entrySet());
        entries.sort(Comparator.comparing(Map.Entry::getKey));
        for (Map.Entry<String, Integer> e : entries) {
            out.writeUTF(e.getKey());
            out.writeInt(e.getValue());
        }
    }

    private static Map<String, Integer> readStringIntMap(DataInputStream in) throws IOException {
        int size = in.readInt();
        Map<String, Integer> map = new LinkedHashMap<>(size);
        for (int i = 0; i < size; i++) {
            map.put(in.readUTF(), in.readInt());
        }
        return map;
    }

    private static void requireNonNegativeLength(int len, String section) throws IOException {
        if (len < 0) {
            throw new IOException("Negative token length in " + section);
        }
    }

    private static byte[] readExactBytes(DataInputStream in, int len, String section) throws IOException {
        byte[] bytes = in.readNBytes(len);
        if (bytes.length != len) {
            throw new IOException("Unexpected EOF while reading " + section);
        }
        return bytes;
    }
}

