package io.github.kirstenali.deepj.quality;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.api.Test;

import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.xpath.XPathFactory;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertEquals;

class BuildRuleScopeTest {

    @ParameterizedTest
    @CsvSource({"maven-checkstyle-plugin,includeTestSourceDirectory", "maven-pmd-plugin,includeTests"})
    void javaRulesIncludeTestsAndTheirHelpers(String plugin, String option) throws Exception {
        String selector = "/project/build/plugins/plugin[artifactId='" + plugin + "']";
        assertEquals("true", pomValue(selector + "/configuration/" + option).trim());
        assertEquals("0", pomValue("count(" + selector + "/executions//" + option + "[.='false'])"));
    }

    @Test
    void normalBuildDoesNotRequireTheFullHardwareProfile() throws Exception {
        assertEquals("--portable", pomValue("/project/properties/coverage.mode").trim());
        assertEquals("0", pomValue("count(/project/profiles/profile[id='full-coverage']/activation)"));
        assertEquals("0", pomValue("count(/project/build/plugins//deepj.requireMetal)"));
    }

    @Test
    void nativeSourceChecksActivateOnlyOnMac() throws Exception {
        String profile = "/project/profiles/profile[id='native-code-quality']";
        assertEquals("mac", pomValue(profile + "/activation/os/family").trim());
        assertEquals("false", pomValue(profile + "/activation/activeByDefault = 'true'"));
    }

    private static String pomValue(String expression) throws Exception {
        DocumentBuilderFactory factory = DocumentBuilderFactory.newInstance();
        factory.setFeature("http://apache.org/xml/features/disallow-doctype-decl", true);
        var document = factory.newDocumentBuilder().parse(Path.of("pom.xml").toFile());
        return XPathFactory.newInstance().newXPath().evaluate(expression, document);
    }
}
