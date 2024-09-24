package uob.oop;

public class HtmlParser {
    /***
     * Extract the title of the news from the _htmlCode.
     * @param _htmlCode Contains the full HTML string from a specific news. E.g. 01.htm.
     * @return Return the title if it's been found. Otherwise, return "Title not found!".
     */
    public static String getNewsTitle(String _htmlCode) {
        String titleTagOpen = "<title>";
        String titleTagClose = "</title>";

        int titleStart = _htmlCode.indexOf(titleTagOpen) + titleTagOpen.length();
        int titleEnd = _htmlCode.indexOf(titleTagClose);

        if (titleStart != -1 && titleEnd != -1 && titleEnd > titleStart) {
            String strFullTitle = _htmlCode.substring(titleStart, titleEnd);
            return strFullTitle.substring(0, strFullTitle.indexOf(" |"));
        }

        return "Title not found!";
    }

    /***
     * Extract the content of the news from the _htmlCode.
     * @param _htmlCode Contains the full HTML string from a specific news. E.g. 01.htm.
     * @return Return the content if it's been found. Otherwise, return "Content not found!".
     */
    public static String getNewsContent(String _htmlCode) {
        String contentTagOpen = "\"articleBody\": \"";
        String contentTagClose = " \",\"mainEntityOfPage\":";

        int contentStart = _htmlCode.indexOf(contentTagOpen) + contentTagOpen.length();
        int contentEnd = _htmlCode.indexOf(contentTagClose);

        if (contentStart != -1 && contentEnd != -1 && contentEnd > contentStart) {
            return _htmlCode.substring(contentStart, contentEnd).toLowerCase();
        }

        return "Content not found!";
    }

    public static NewsArticles.DataType getDataType(String _htmlCode) {
        final String dataTypeTagOpen = "<datatype>";
        final String dataTypeTagClose = "</datatype>";

        int dataTypeStart = _htmlCode.indexOf(dataTypeTagOpen) + dataTypeTagOpen.length();
        int dataTypeEnd = _htmlCode.indexOf(dataTypeTagClose);

        if (dataTypeStart != -1 && dataTypeEnd != -1 && dataTypeEnd > dataTypeStart) {
            String dataType = _htmlCode.substring(dataTypeStart, dataTypeEnd);
            if (dataType.equalsIgnoreCase("training")) {
                return NewsArticles.DataType.Training;
            } /*else if (dataType.equalsIgnoreCase("testing")) {
                return NewsArticles.DataType.Testing;
            } */
        }

        return NewsArticles.DataType.Testing; // Default case
    }


    public static String getLabel (String _htmlCode) {
        final String labelOpen = "<label>";
        final String labelClose = "</label>";

        int labelStart = _htmlCode.indexOf(labelOpen) + labelOpen.length();
        int labelEnd = _htmlCode.indexOf(labelClose);

        if (labelStart != -1 && labelEnd != -1 && labelEnd > labelStart) {
            return _htmlCode.substring(labelStart, labelEnd);
        }

        return "-1"; // If no label tag found
    }


}
