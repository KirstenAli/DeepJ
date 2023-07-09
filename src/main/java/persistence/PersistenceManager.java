package persistence;

import com.fasterxml.jackson.databind.MapperFeature;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.json.JsonMapper;

import java.io.File;
import java.io.IOException;


public class PersistenceManager {

    public static void saveObjectAsJson(Object object, String filePath, String fileName) {
        // Create an ObjectMapper instance
        ObjectMapper objectMapper = JsonMapper
                .builder()
                .disable(MapperFeature.AUTO_DETECT_GETTERS)
                .build();

        try {
            // Convert the object to JSON and write it to a file
            objectMapper.writeValue(new File(fileName), object);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
