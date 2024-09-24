package uob.oop;

import org.apache.commons.lang3.time.StopWatch;
import org.deeplearning4j.datasets.iterator.utilty.ListDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class AdvancedNewsClassifier {
    public Toolkit myTK = null;
    public static List<NewsArticles> listNews = null;
    public static List<Glove> listGlove = null;
    public List<ArticlesEmbedding> listEmbedding = null;
    public MultiLayerNetwork myNeuralNetwork = null;

    public final int BATCHSIZE = 10;

    public int embeddingSize = 0;
    private static StopWatch mySW = new StopWatch();

    public AdvancedNewsClassifier() throws IOException {
        myTK = new Toolkit();
        myTK.loadGlove();
        listNews = myTK.loadNews();
        listGlove = createGloveList();
        listEmbedding = loadData();
    }

    public static void main(String[] args) throws Exception {
        mySW.start();
        AdvancedNewsClassifier myANC = new AdvancedNewsClassifier();

        myANC.embeddingSize = myANC.calculateEmbeddingSize(myANC.listEmbedding);
        myANC.populateEmbedding();
        myANC.myNeuralNetwork = myANC.buildNeuralNetwork(2);
        myANC.predictResult(myANC.listEmbedding);
        myANC.printResults();
        mySW.stop();
        System.out.println("Total elapsed time: " + mySW.getTime());
    }

    public List<Glove> createGloveList() {
        List<Glove> listResult = new ArrayList<>();

        for (int i = 0; i < Toolkit.listVocabulary.size(); i++) {
            String word = Toolkit.listVocabulary.get(i);

            if (!ArticlesEmbedding.isStopword(word)) {
                double[] vectorElements = Toolkit.listVectors.get(i);
                Vector vector = new Vector(vectorElements);
                Glove glove = new Glove(word, vector);
                listResult.add(glove);
            }
        }

        return listResult;
    }


    public static List<ArticlesEmbedding> loadData() {
        List<ArticlesEmbedding> listEmbedding = new ArrayList<>();
        for (NewsArticles news : listNews) {
            ArticlesEmbedding myAE = new ArticlesEmbedding(news.getNewsTitle(), news.getNewsContent(), news.getNewsType(), news.getNewsLabel());
            listEmbedding.add(myAE);
        }
        return listEmbedding;
    }

    private static void merge(List<Integer> arr, int l, int m, int r) { //merge
        int n1 = m - l + 1;
        int n2 = r - m;

        List<Integer> L = new ArrayList<>(arr.subList(l, l + n1));
        List<Integer> R = new ArrayList<>(arr.subList(m + 1, m + 1 + n2));


        int i = 0, j = 0, k = l;
        while (i < n1 && j < n2) {
            if (L.get(i) <= R.get(j)) {
                arr.set(k, L.get(i));
                i++;
            } else {
                arr.set(k, R.get(j));
                j++;
            }
            k++;
        }

        while (i < n1) {
            arr.set(k, L.get(i));
            i++;
            k++;
        }

        while (j < n2) {
            arr.set(k, R.get(j));
            j++;
            k++;
        }
    }

    private static void mergeSort(List<Integer> arr, int l, int r) {
        if (l < r) {
            int m = (l + r) / 2;

            mergeSort(arr, l, m);
            mergeSort(arr, m + 1, r);

            merge(arr, l, m, r);
        }
    }

    private void quickSort(int[] arr, int low, int high) {
        if (low < high) {
            int partitionIndex = partition(arr, low, high);

            quickSort(arr, low, partitionIndex - 1);
            quickSort(arr, partitionIndex + 1, high);
        }
    }

    private int partition(int[] arr, int low, int high) {
        int pivot = arr[high];
        int i = low - 1;

        for (int j = low; j < high; j++) {
            if (arr[j] <= pivot) {
                i++;

                int temp = arr[i];
                arr[i] = arr[j];
                arr[j] = temp;
            }
        }

        int temp = arr[i + 1];
        arr[i + 1] = arr[high];
        arr[high] = temp;

        return i + 1;
    }

    public int calculateEmbeddingSize(List<ArticlesEmbedding> _listEmbedding) {
        int intMedian = -1;
        int[] lengths = new int[_listEmbedding.size()];
        for (int i = 0; i < _listEmbedding.size(); i++) {
            lengths[i] = _listEmbedding.get(i).getWordCount(Toolkit.getListVocabulary());
            // System.out.print(documentLengths[i] + " ");
        }

        quickSort(lengths, 0, lengths.length - 1);

        /* for (int i: lengths) {
            System.out.print(i + ", ");
        } */

        int n = lengths.length;

        if (n % 2 == 0) {
            intMedian = (lengths[n/2] + lengths[(n/2)+1]) / 2;
        } else intMedian = lengths[(n+1)/2];

        return intMedian;
    }

    public void populateEmbedding() {
        for (ArticlesEmbedding articlesEmbedding : listEmbedding) {
            boolean populate = true;
            while (populate) {
                try {
                    articlesEmbedding.getEmbedding();
                    populate = false;
                } catch (InvalidSizeException e) {
                    articlesEmbedding.setEmbeddingSize(embeddingSize);
                } catch (InvalidTextException e) {
                    articlesEmbedding.getNewsContent();
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        }
    }


    public DataSetIterator populateRecordReaders(int _numberOfClasses) throws Exception {
        ListDataSetIterator myDataIterator = null;
        List<DataSet> listDS = new ArrayList<>();
        INDArray inputNDArray = null;
        INDArray outputNDArray = null;

        for (ArticlesEmbedding articlesEmbedding : listEmbedding) {
            if (articlesEmbedding.getNewsType() != NewsArticles.DataType.Training) {
                continue;
            }

            inputNDArray = articlesEmbedding.getEmbedding();
            outputNDArray = Nd4j.zeros(1, _numberOfClasses);
            int i = Integer.parseInt(articlesEmbedding.getNewsLabel()) - 1;
            outputNDArray.putScalar(new int[] {0, i}, 1);
            listDS.add(new DataSet(inputNDArray, outputNDArray));

        }

        return new ListDataSetIterator<>(listDS, BATCHSIZE);
    }

    public MultiLayerNetwork buildNeuralNetwork(int _numOfClasses) throws Exception {
        DataSetIterator trainIter = populateRecordReaders(_numOfClasses);
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(42)
                .trainingWorkspaceMode(WorkspaceMode.ENABLED)
                .activation(Activation.RELU)
                .weightInit(WeightInit.XAVIER)
                .updater(Adam.builder().learningRate(0.02).beta1(0.9).beta2(0.999).build())
                .l2(1e-4)
                .list()
                .layer(new DenseLayer.Builder().nIn(embeddingSize).nOut(15)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.HINGE)
                        .activation(Activation.SOFTMAX)
                        .nIn(15).nOut(_numOfClasses).build())
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        for (int n = 0; n < 100; n++) {
            model.fit(trainIter);
            trainIter.reset();
        }
        return model;
    }

    public List<Integer> predictResult(List<ArticlesEmbedding> _listEmbedding) throws Exception {
        List<Integer> listResult = new ArrayList<>();
        for (ArticlesEmbedding articlesEmbedding : _listEmbedding) {
            if (articlesEmbedding.getNewsType() != NewsArticles.DataType.Testing) {
                continue;
            }

            int pred = myNeuralNetwork.predict(articlesEmbedding.getEmbedding())[0];
            articlesEmbedding.setNewsLabel(Integer.toString(pred + 1));
            listResult.add(pred);
        }

        return listResult;
    }

    public void printResults() {
        int max = 0;
        for (ArticlesEmbedding articlesEmbedding : listEmbedding) {
            if (articlesEmbedding.getNewsType() != NewsArticles.DataType.Testing) {
                continue;
            }

            int label = Integer.parseInt(articlesEmbedding.getNewsLabel());
            if (max < label) {
                max = label;
            }
        }

        List<List<String>> listArr = new ArrayList<>();
        for (int i = 0; i < max; i++) {
            listArr.add(new ArrayList<>());
        }

        for (ArticlesEmbedding articlesEmbedding : listEmbedding) {
            if (articlesEmbedding.getNewsType() != NewsArticles.DataType.Testing) {
                continue;
            }

            String title = articlesEmbedding.getNewsTitle();
            int label = Integer.parseInt(articlesEmbedding.getNewsLabel());
            listArr.get(label - 1).add(title);
        }

        StringBuilder result = new StringBuilder();

        for (int i = 0; i < listArr.size(); i++) {
            result.append("Group ");
            result.append(i + 1);
            result.append("\r\n");
            for (String title : listArr.get(i)) {
                result.append(title);
                result.append("\r\n");
            }
        }
        System.out.print(result);
    }
}
