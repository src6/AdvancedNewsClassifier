package uob.oop;

public class NewsArticles {
    public enum DataType {
        Training, Testing
    }

    private String newsTitle = "", newsContent = "";
    private DataType newsType = DataType.Testing;
    private String newsLabel = "-1";

    public NewsArticles(String _title, String _content, DataType _type, String _label) {
        this.newsTitle = _title;
        this.newsContent = _content;
        this.newsType = _type;
        this.newsLabel = _label;
    }

    public String getNewsLabel() {
        return newsLabel;
    }

    public DataType getNewsType() {
        return newsType;
    }

    public String getNewsTitle() {
        return newsTitle;
    }

    public String getNewsContent() {
        return newsContent;
    }

    public void setNewsLabel(String _lable) {
        this.newsLabel = _lable;
    }

    public void setNewsType (DataType _type){
        this.newsType = _type;
    }
}
