package uob.oop;

import edu.stanford.nlp.ling.*;
import edu.stanford.nlp.pipeline.*;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.List;
import java.util.Properties;


public class ArticlesEmbedding extends NewsArticles {
    private int intSize = -1;
    private String processedText = "";

    private INDArray newsEmbedding = Nd4j.create(0);

    public ArticlesEmbedding(String _title, String _content, NewsArticles.DataType _type, String _label) {
        super(_title,_content,_type,_label);
    }

    public void setEmbeddingSize(int _size) {
        this.intSize = _size;
    }

    public int getEmbeddingSize(){
        return intSize;
    }

    private String lemmatizeText(String text) {
        Properties props = new Properties();
        props.setProperty("annotators", "tokenize,pos,lemma");
        StanfordCoreNLP pipeline = new StanfordCoreNLP(props);

        CoreDocument document = new CoreDocument(text);
        pipeline.annotate(document);

        StringBuilder lemmatizedText = new StringBuilder();
        for (CoreLabel token : document.tokens()) {
            lemmatizedText.append(token.lemma()).append(" ");
        }

        return lemmatizedText.toString().trim();
    }

    public static boolean isStopword(String word) {
        for (String stopWord : Toolkit.STOPWORDS) {
            if (word.equalsIgnoreCase(stopWord)) {
                return true;
            }
        }
        return false;
    }

    private String removeStopWords(String text) {
        String[] words = text.split("\\s+");
        StringBuilder result = new StringBuilder();

        for (String word : words) {
            if (!isStopword(word)) {
                result.append(word).append(" ");
            }
        }

        return result.toString().trim();
    }

    @Override
    public String getNewsContent() {
        if (processedText.isEmpty()) {
            String cleanedText = textCleaning(super.getNewsContent());
            String lemmatizedText = lemmatizeText(cleanedText);
            processedText = removeStopWords(lemmatizedText).toLowerCase();
        }

        return processedText.trim();
    }

    public INDArray getEmbedding() throws Exception {
        if (intSize == -1) {
            throw new InvalidSizeException("Invalid size");
        }

        if (processedText.isEmpty()) {
            throw new InvalidTextException("Invalid text");
        }

        if (newsEmbedding.isEmpty()) {
            AdvancedNewsClassifier myANC = new AdvancedNewsClassifier();
            List<Glove> gloveList = myANC.createGloveList();

            INDArray wordEmbeddings = Nd4j.create(intSize, gloveList.get(0).getVector().getVectorSize());

            String[] words = processedText.split("\\s+");
            int row = 0;

            for (String word : words) {
                if (row >= intSize) {
                    break;
                }

                Glove glove = gloveList.stream()
                        .filter(g -> g.getVocabulary().equals(word))
                        .findFirst()
                        .orElse(null);

                if (glove != null) {
                    wordEmbeddings.putRow(row, Nd4j.create(glove.getVector().getAllElements()));
                    row++;
                }
            }

            newsEmbedding = wordEmbeddings;
        }

        return Nd4j.vstack(newsEmbedding.mean(1));
    }

    /***
     * Clean the given (_content) text by removing all the characters that are not 'a'-'z', '0'-'9' and white space.
     * @param _content Text that need to be cleaned.
     * @return The cleaned text.
     */
    private static String textCleaning(String _content) {
        StringBuilder sbContent = new StringBuilder();

        for (char c : _content.toLowerCase().toCharArray()) {
            if ((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || Character.isWhitespace(c)) {
                sbContent.append(c);
            }
        }

        return sbContent.toString().trim();
    }

    public int getWordCount(List<String> listVocabulary) {
        String content = getNewsContent();
        String[] words = content.split("\\s+");

        int count = 0;
        for (String word : words) {
            if (listVocabulary.contains(word)) {
                count++;
            }
        }

        return count;
    }

}


