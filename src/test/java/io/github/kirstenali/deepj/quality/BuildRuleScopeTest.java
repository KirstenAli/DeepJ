package io.github.kirstenali.deepj.quality;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.xpath.XPathFactory;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertEquals;

class BuildRuleScopeTest {

    @ParameterizedTest
    @CsvSource({"maven-checkstyle-plugin,includeTestSourceDirectory", "maven-pmd-plugin,includeTests"})
    void javaRulesIncludeTestsAndTheirHelpers(String plugin, String option) throws Exception {
        DocumentBuilderFactory factory = DocumentBuilderFactory.newInstance();
        factory.setFeature("http://apache.org/xml/features/disallow-doctype-decl", true);
        var document = factory.newDocumentBuilder().parse(Path.of("pom.xml").toFile());
        var xpath = XPathFactory.newInstance().newXPath();
        String selector = "/project/build/plugins/plugin[artifactId='" + plugin + "']";
        assertEquals("true", xpath.evaluate(selector + "/configuration/" + option, document).trim());
        assertEquals("0", xpath.evaluate("count(" + selector + "/executions//" + option + "[.='false'])", document));
    }
}
