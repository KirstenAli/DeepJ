package persistence;

import com.fasterxml.jackson.databind.MapperFeature;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.json.JsonMapper;
import org.jbackprop.ann.Network;

import java.io.*;

public class PersistenceManager {

    private static void saveObjectAsJson(Object object,
                                         String filePath,
                                         ObjectMapper objectMapper) {
        try {
            objectMapper.writeValue(new File(filePath), object);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static void serializeObject(Object object, String filePath) {
        try (FileOutputStream fileOut = new FileOutputStream(filePath);
             ObjectOutputStream objectOut = new ObjectOutputStream(fileOut)) {
            objectOut.writeObject(object);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static Object deserializeObject(String filePath) {
        try (FileInputStream fileIn = new FileInputStream(filePath);
             ObjectInputStream objectIn = new ObjectInputStream(fileIn)) {
            return objectIn.readObject();
        } catch (IOException | ClassNotFoundException e) {
            e.printStackTrace();
        }
        return null;
    }

    public static void saveWeightsAsJson(Object network,
                                         String filePath) {

        ObjectMapper objectMapper = JsonMapper
                .builder()
                .disable(MapperFeature.AUTO_DETECT_GETTERS)
                .build();

        saveObjectAsJson(network, filePath, objectMapper);
    }

    public static void saveNetwork(Object network, String filePath) {
        serializeObject(network, filePath);
    }

    public static Network loadNetwork(String filePath) {
        return (Network) deserializeObject(filePath);
    }
}
