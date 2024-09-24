package uob.oop;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.net.URISyntaxException;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Stream;

public class Toolkit {
    public static List<String> listVocabulary = null;
    public static List<double[]> listVectors = null;
    private static final String FILENAME_GLOVE = "glove.6B.50d_Reduced.csv";

    public static final String[] STOPWORDS = {"a", "able", "about", "across", "after", "all", "almost", "also", "am", "among", "an", "and", "any", "are", "as", "at", "be", "because", "been", "but", "by", "can", "cannot", "could", "dear", "did", "do", "does", "either", "else", "ever", "every", "for", "from", "get", "got", "had", "has", "have", "he", "her", "hers", "him", "his", "how", "however", "i", "if", "in", "into", "is", "it", "its", "just", "least", "let", "like", "likely", "may", "me", "might", "most", "must", "my", "neither", "no", "nor", "not", "of", "off", "often", "on", "only", "or", "other", "our", "own", "rather", "said", "say", "says", "she", "should", "since", "so", "some", "than", "that", "the", "their", "them", "then", "there", "these", "they", "this", "tis", "to", "too", "twas", "us", "wants", "was", "we", "were", "what", "when", "where", "which", "while", "who", "whom", "why", "will", "with", "would", "yet", "you", "your"};

    public void loadGlove() throws IOException {
        BufferedReader myReader = null;
        listVocabulary = new ArrayList<>();
        listVectors = new ArrayList<>();

        try {
            myReader = new BufferedReader(new FileReader(Toolkit.getFileFromResource(FILENAME_GLOVE)));
            String line;
            while ((line = myReader.readLine()) != null) {
                String[] lineArr = line.split(",");
                String word = lineArr[0].trim();

                listVocabulary.add(word);
                double[] tempVector = new double[lineArr.length-1];
                for(int i=1; i< lineArr.length; i++) {
                    tempVector[i-1] = Double.parseDouble(lineArr[i].trim());
                }
                listVectors.add(tempVector);
            }
        } catch (Exception e) {
            System.err.println(e.getMessage());
        } finally {
            if (myReader != null) {
                try {
                    myReader.close();
                } catch (IOException e) {
                    System.err.println(e);
                }
            }
        }
    }


    private static File getFileFromResource(String fileName) throws URISyntaxException {
        ClassLoader classLoader = Toolkit.class.getClassLoader();
        URL resource = classLoader.getResource(fileName);
        if (resource == null) {
            throw new IllegalArgumentException(fileName);
        } else {
            return new File(resource.toURI());
        }
    }

    public List<NewsArticles> loadNews() {
        List<NewsArticles> listNews = new ArrayList<>();
        try {
            File newsFolder = getFileFromResource("News");

            List<File> htmlFiles = Files.list(newsFolder.toPath())
                    .map(Path::toFile)
                    .filter(file -> file.getName().toLowerCase().endsWith(".htm"))
                    .sorted(Comparator.naturalOrder())
                    .toList();

            for (File htmlFile : htmlFiles) {
                String htmlContent = Files.readString(htmlFile.toPath());

                String title = HtmlParser.getNewsTitle(htmlContent);
                String content = HtmlParser.getNewsContent(htmlContent);
                NewsArticles.DataType dataType = HtmlParser.getDataType(htmlContent);
                String label = HtmlParser.getLabel(htmlContent);

                NewsArticles newsArticle = new NewsArticles(title, content, dataType, label);
                listNews.add(newsArticle);
            }
        } catch (IOException | URISyntaxException e) {
            System.err.println(e.getMessage());
        }

        return listNews;
    }

    public static List<String> getListVocabulary() {
        return listVocabulary;
    }

    public static List<double[]> getlistVectors() {
        return listVectors;
    }
}
