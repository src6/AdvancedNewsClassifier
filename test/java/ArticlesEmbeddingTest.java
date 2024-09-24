import org.apache.commons.lang3.time.StopWatch;
import org.junit.jupiter.api.Test;
import uob.oop.*;

import java.io.IOException;

import static org.junit.Assert.assertThrows;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class ArticlesEmbeddingTest {
    private ArticlesEmbedding myEmbedding = new ArticlesEmbedding("What is COVID-19?", "Coronavirus disease 2019 (COVID-19) is a contagious disease caused by the virus SARS-CoV-2. The first known case was identified in Wuhan, China, in December 2019.[6] The disease quickly spread worldwide, resulting in the COVID-19 pandemic.\n" +
            "\n" +
            "The symptoms of COVID‑19 are variable but often include fever,[7] cough, headache,[8] fatigue, breathing difficulties, loss of smell, and loss of taste.[9][10][11] Symptoms may begin one to fourteen days after exposure to the virus. At least a third of people who are infected do not develop noticeable symptoms.[12][13] Of those who develop symptoms noticeable enough to be classified as patients, most (81%) develop mild to moderate symptoms (up to mild pneumonia), while 14% develop severe symptoms (dyspnea, hypoxia, or more than 50% lung involvement on imaging), and 5% develop critical symptoms (respiratory failure, shock, or multiorgan dysfunction).[14] Older people are at a higher risk of developing severe symptoms. Some people continue to experience a range of effects (long COVID) for years after infection, and damage to organs has been observed.[15] Multi-year studies are underway to further investigate the long-term effects of the disease.[16]\n" +
            "\n" +
            "COVID‑19 transmits when infectious particles are breathed in or come into contact with the eyes, nose, or mouth. The risk is highest when people are in close proximity, but small airborne particles containing the virus can remain suspended in the air and travel over longer distances, particularly indoors. Transmission can also occur when people touch their eyes, nose or mouth after touching surfaces or objects that have been contaminated by the virus. People remain contagious for up to 20 days and can spread the virus even if they do not develop symptoms.[17]\n" +
            "\n" +
            "Testing methods for COVID-19 to detect the virus's nucleic acid include real-time reverse transcription polymerase chain reaction (RT‑PCR),[18][19] transcription-mediated amplification,[18][19][20] and reverse transcription loop-mediated isothermal amplification (RT‑LAMP)[18][19] from a nasopharyngeal swab.[21]\n" +
            "\n" +
            "Several COVID-19 vaccines have been approved and distributed in various countries, which have initiated mass vaccination campaigns. Other preventive measures include physical or social distancing, quarantining, ventilation of indoor spaces, use of face masks or coverings in public, covering coughs and sneezes, hand washing, and keeping unwashed hands away from the face. While work is underway to develop drugs that inhibit the virus, the primary treatment is symptomatic. Management involves the treatment of symptoms through supportive care, isolation, and experimental measures. ", NewsArticles.DataType.Training, "1");
    private StopWatch mySW = new StopWatch();

    @Test
    void ArticlesEmbeddingConstructor() {
        assertEquals("What is COVID-19?", myEmbedding.getNewsTitle());
        assertEquals(NewsArticles.DataType.Training, myEmbedding.getNewsType());
    }

    @Test
    void setEmbeddingSize() {
        myEmbedding.setEmbeddingSize(10);
        assertEquals(10, myEmbedding.getEmbeddingSize());
    }

    @Test
    void getNewsContent_Functional() {
        assertEquals("coronavirus disease 2019 covid19 contagious disease cause virus sarscov2 first know case identify wuhan china december 20196 disease quickly spread worldwide result covid19 pandemic symptom covid19 variable include fever7 cough headache8 fatigue breathing difficulty loss smell loss taste91011 symptom begin one fourteen day exposure virus third person infect develop noticeable symptoms1213 develop symptom noticeable enough classify patient 81 develop mild moderate symptom up mild pneumonia 14 develop severe symptom dyspnea hypoxia more 50 lung involvement imaging 5 develop critical symptom respiratory failure shock multiorgan dysfunction14 old person high risk develop severe symptom person continue experience range effect long covid year infection damage organ observed15 multiyear study underway further investigate longterm effect disease16 covid19 transmit infectious particle breathe come contact eye nose mouth risk high person close proximity small airborne particle contain virus remain suspend air travel over long distance particularly indoors transmission occur person touch eye nose mouth touch surface object contaminate virus person remain contagious up 20 day spread virus even develop symptoms17 testing method covid19 detect viruss nucleic acid include realtime reverse transcription polymerase chain reaction rtpcr1819 transcriptionmediate amplification181920 reverse transcription loopmediate isothermal amplification rtlamp1819 nasopharyngeal swab21 several covid19 vaccine approve distribute various country initiate mass vaccination campaign preventive measure include physical social distancing quarantine ventilation indoor space use face mask covering public covering cough sneeze hand washing keep unwashed hand away face work underway develop drug inhibit virus primary treatment symptomatic management involve treatment symptom through supportive care isolation experimental measure", myEmbedding.getNewsContent());
    }

    @Test
    void getNewsContent_Performance() {
        long totalTime = 0;
        for (int i = 0; i < 100; i++) {
            mySW.start();
            myEmbedding.getNewsContent();
            mySW.stop();
            totalTime += mySW.getNanoTime();
            mySW.reset();
        }
        System.out.println("Average execution time: " + (totalTime / 100));
        assertTrue(totalTime / 100 < 13000000);
    }

    @Test
    void getEmbedding_Functional() throws Exception {
        AdvancedNewsClassifier myANC = new AdvancedNewsClassifier();
        assertThrows(InvalidSizeException.class, () -> {
            myEmbedding.getEmbedding();
        });
        myEmbedding.setEmbeddingSize(100);
        assertThrows(InvalidTextException.class, () -> {
            myEmbedding.getEmbedding();
        });
        myEmbedding.getNewsContent();
        assertEquals("[[    0.1972,    0.1972,    0.1096,    0.1971,    0.1129,    0.0215,    0.0684,    0.0427,   -0.1385,    0.1972,    0.0726,    0.0459,    0.0961,    0.1729,    0.2335,    0.1985,    0.1027,    0.2680,    0.1357,    0.0138,    0.1473,    0.0138,    0.1729,    0.0171,   -0.0441,    0.0148,    0.1865,    0.1971,   -0.0170,    0.1991,    0.1729,    0.1335,    0.2661,    0.0136,    0.1729,    0.2547,    0.1729,    0.2259,    0.2719,    0.1866,   -0.0467,    0.1513,    0.1729,    0.0634,    0.1042,    0.0422,    0.1991,    0.0183,    0.1948,    0.1729,    0.1991,    0.1081,    0.0752,    0.0386,    0.2290,    0.1108,    0.1088,    0.0446,    0.0386,    0.1255,    0.0412,    0.1745,    0.1245,    0.0509,    0.0816,    0.1948,    0.0183,    0.1991,    0.0343,    0.1021,    0.0576,    0.1255,    0.1971,    0.0472,    0.1228,    0.0773,    0.2124,    0.1991,    0.1261,    0.1245,    0.0509,    0.0816,    0.1261,    0.1658,    0.0638,    0.1971,    0.1991,    0.0148,    0.0726,    0.1971,    0.0268,    0.1022,    0.1447,    0.3049,   -0.0328,    0.2135,    0.1759,    0.0761,    0.1239,   -0.0328]]", myEmbedding.getEmbedding().toString());
    }

    @Test
    void getEmbedding_Performance() throws Exception {
        long totalTime = 0;
        for (int i = 0; i < 50; i++) {
            AdvancedNewsClassifier myANC = new AdvancedNewsClassifier();
            myEmbedding.setEmbeddingSize(100);
            myEmbedding.getNewsContent();
            mySW.start();
            myEmbedding.getEmbedding();
            mySW.stop();
            totalTime += mySW.getTime();
            mySW.reset();
        }
        System.out.println("Average execution time: " + (totalTime / 50));
        assertTrue(totalTime / 50 < 8);
    }
}
