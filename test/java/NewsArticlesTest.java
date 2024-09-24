import org.junit.jupiter.api.Test;
import uob.oop.NewsArticles;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class NewsArticlesTest {
    @Test
    void newsTest() {
        NewsArticles myArticles = new NewsArticles("Hi", "Hello world!", NewsArticles.DataType.Training, "-1");
        assertEquals("Hi", myArticles.getNewsTitle());
        assertEquals("Hello world!", myArticles.getNewsContent());
        assertEquals(NewsArticles.DataType.Training, myArticles.getNewsType());
        assertEquals("-1", myArticles.getNewsLabel());
    }
}
