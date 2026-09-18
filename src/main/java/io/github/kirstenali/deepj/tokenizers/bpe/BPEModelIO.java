package io.github.kirstenali.deepj.tokenizers.bpe;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public final class BPEModelIO {

    private static final int MAGIC = 0x444A4250; // DJBP
    private static final int MAX_COLLECTION_SIZE = 1_000_000;
    private static final int MAX_TOKEN_BYTES = 16 * 1024 * 1024;

    private BPEModelIO() {
    }

    public static void save(Path path, BPEModel model) throws IOException {
        ensureParentDirectory(path);
        Path temporary = temporaryPath(path);
        try {
            writeModel(temporary, model);
            replaceAtomically(temporary, path);
        } finally {
            Files.deleteIfExists(temporary);
        }
    }

    private static void writeModel(Path path, BPEModel model) throws IOException {
        try (DataOutputStream out = new DataOutputStream(new BufferedOutputStream(Files.newOutputStream(path)))) {
            out.writeInt(MAGIC);
            out.writeInt(model.modelFormatVersion());
            out.writeInt(model.endOfWordId());

            writeVocab(out, model.idToBytes());
            writeStringIntMap(out, model.tokenKeyToId());
            writeMerges(out, model.merges());
            writeMergeToNewId(out, model.mergeToNewId());
            writeStringIntMap(out, model.specialTokenToId());
        }
    }

    private static Path temporaryPath(Path path) throws IOException {
        Path absolute = path.toAbsolutePath();
        return Files.createTempFile(absolute.getParent(), absolute.getFileName().toString(), ".tmp");
    }

    private static void replaceAtomically(Path temporary, Path destination) throws IOException {
        try {
            Files.move(temporary, destination, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING);
        } catch (AtomicMoveNotSupportedException e) {
            Files.move(temporary, destination, StandardCopyOption.REPLACE_EXISTING);
        }
    }

    public static BPEModel load(Path path) throws IOException {
        try (DataInputStream in = new DataInputStream(new BufferedInputStream(Files.newInputStream(path)))) {
            return readModel(in);
        } catch (IllegalArgumentException e) {
            throw new IOException("Invalid BPE model structure", e);
        }
    }

    private static BPEModel readModel(DataInputStream in) throws IOException {
        requireMagic(in.readInt());
        int version = requireVersion(in.readInt());
        int endOfWordId = in.readInt();
        List<byte[]> vocab = readVocab(in);
        Map<String, Integer> tokenIds = readStringIntMap(in);
        List<TokenPair> merges = readMerges(in);
        Map<TokenPair, Integer> mergeIds = readMergeToNewId(in);
        Map<String, Integer> specials = readStringIntMap(in);
        BPEModel model = new BPEModel(vocab, tokenIds, merges, mergeIds, endOfWordId, version, specials);
        if (in.read() != -1) throw new IOException("Unexpected trailing BPE model data");
        return model;
    }

    private static void requireMagic(int magic) throws IOException {
        if (magic != MAGIC) throw new IOException("Invalid BPE model file magic");
    }

    private static int requireVersion(int version) throws IOException {
        if (version != BPEModel.CURRENT_FORMAT_VERSION) {
            throw new IOException("Unsupported BPE model version: " + version);
        }
        return version;
    }

    private static void ensureParentDirectory(Path path) throws IOException {
        Path parent = path.getParent();
        if (parent != null) Files.createDirectories(parent);
    }

    private static void writeVocab(DataOutputStream out, List<byte[]> vocab) throws IOException {
        out.writeInt(vocab.size());
        for (byte[] token : vocab) {
            out.writeInt(token.length);
            out.write(token);
        }
    }

    private static List<byte[]> readVocab(DataInputStream in) throws IOException {
        int size = readCollectionSize(in, "vocabulary");
        List<byte[]> vocab = new ArrayList<>(size);
        for (int i = 0; i < size; i++) {
            int len = in.readInt();
            requireTokenLength(len);
            byte[] token = readExactBytes(in, len);
            vocab.add(token);
        }
        return vocab;
    }


    private static void writeMerges(DataOutputStream out, List<TokenPair> merges) throws IOException {
        out.writeInt(merges.size());
        for (TokenPair pair : merges) {
            out.writeInt(pair.left());
            out.writeInt(pair.right());
        }
    }

    private static List<TokenPair> readMerges(DataInputStream in) throws IOException {
        int size = readCollectionSize(in, "merges");
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
        int size = readCollectionSize(in, "merge map");
        Map<TokenPair, Integer> map = new LinkedHashMap<>(size);
        for (int i = 0; i < size; i++) {
            TokenPair pair = new TokenPair(in.readInt(), in.readInt());
            map.put(pair, in.readInt());
        }
        return map;
    }


    private static void writeStringIntMap(DataOutputStream out, Map<String, Integer> map) throws IOException {
        out.writeInt(map.size());
        List<Map.Entry<String, Integer>> entries = new ArrayList<>(map.entrySet());
        entries.sort(Map.Entry.comparingByKey());
        for (Map.Entry<String, Integer> e : entries) {
            out.writeUTF(e.getKey());
            out.writeInt(e.getValue());
        }
    }

    private static Map<String, Integer> readStringIntMap(DataInputStream in) throws IOException {
        int size = readCollectionSize(in, "string map");
        Map<String, Integer> map = new LinkedHashMap<>(size);
        for (int i = 0; i < size; i++) {
            map.put(in.readUTF(), in.readInt());
        }
        return map;
    }

    private static int readCollectionSize(DataInputStream in, String label) throws IOException {
        int size = in.readInt();
        if (size < 0 || size > MAX_COLLECTION_SIZE) throw new IOException("Invalid " + label + " size: " + size);
        return size;
    }

    private static void requireTokenLength(int length) throws IOException {
        if (length < 0 || length > MAX_TOKEN_BYTES) throw new IOException("Invalid token length: " + length);
    }

    private static byte[] readExactBytes(DataInputStream in, int len) throws IOException {
        byte[] bytes = in.readNBytes(len);
        if (bytes.length != len) {
            throw new IOException("Unexpected EOF while reading vocab bytes");
        }
        return bytes;
    }
}
