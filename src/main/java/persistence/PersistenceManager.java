package persistence;

import com.fasterxml.jackson.databind.MapperFeature;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.json.JsonMapper;

import java.io.File;
import java.io.IOException;

public class PersistenceManager {

    private static void saveObjectAsJson(Object object,
                                        String filePath,
                                        String fileName,
                                        ObjectMapper objectMapper) {
        try {
            objectMapper.writeValue(new File(fileName), object);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void saveWeightsAsJson(Object object,
                                         String filePath,
                                         String fileName) {

        ObjectMapper objectMapper = JsonMapper
                .builder()
                .disable(MapperFeature.AUTO_DETECT_GETTERS)
                .build();

        saveObjectAsJson(object, filePath, fileName, objectMapper);
    }
}
