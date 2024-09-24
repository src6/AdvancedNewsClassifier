import org.junit.jupiter.api.Test;
import uob.oop.HtmlParser;
import uob.oop.NewsArticles;
import uob.oop.Toolkit;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class HtmlParserTest {
    private String strHTML = "<datatype>Training</datatype><label>2</label>";
    private String strHTML2 = "<datatype>Testing</datatype><label>1</label>";

    @Test
    void getDataType() {
        assertEquals(NewsArticles.DataType.Training, HtmlParser.getDataType(strHTML));
        assertEquals(NewsArticles.DataType.Testing, HtmlParser.getDataType(strHTML2));
    }

    @Test
    void getLabel() {
        assertEquals("2", HtmlParser.getLabel(strHTML));
        assertEquals("1", HtmlParser.getLabel(strHTML2));
    }
}
