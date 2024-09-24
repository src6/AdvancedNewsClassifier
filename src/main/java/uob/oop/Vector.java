package uob.oop;

public class Vector {
    private double[] doubElements;

    public Vector(double[] _elements) {
        doubElements = _elements;
    }

    public double getElementatIndex(int _index) {
        if (_index >= 0 && _index < doubElements.length)
            return doubElements[_index];
        else
            return -1;
    }

    public void setElementatIndex(double _value, int _index) {
        if (_index >= 0 && _index < doubElements.length)
            doubElements[_index] = _value;
        else
            doubElements[doubElements.length - 1] = _value;
    }

    public double[] getAllElements() {
        return doubElements;
    }

    public int getVectorSize() {
        return doubElements.length;
    }

    public Vector reSize(int _size) {
        if (this.getVectorSize() == _size || _size <= 0)
            return this;

        double[] doubElementNew = new double[_size];

        for (int i = 0; i < (Math.min(this.getVectorSize(), _size)); i++) {
            doubElementNew[i] = this.getElementatIndex(i);
        }

        for (int i = this.getVectorSize(); i < _size; i++) {
            doubElementNew[i] = -1.0;
        }

        return new Vector(doubElementNew);
    }

    public Vector add(Vector _v) {
        Vector myVector = this;
        if (this.getVectorSize() > _v.getVectorSize())
            _v = _v.reSize(this.getVectorSize());
        else if (this.getVectorSize() < _v.getVectorSize())
            myVector = this.reSize(_v.getVectorSize());

        double[] doubElementNew = new double[_v.getVectorSize()];

        for (int i = 0; i < _v.getVectorSize(); i++) {
            doubElementNew[i] = myVector.getElementatIndex(i) + _v.getElementatIndex(i);
        }
        return new Vector(doubElementNew);
    }

    public Vector subtraction(Vector _v) {
        Vector myVector = this;
        if (this.getVectorSize() > _v.getVectorSize())
            _v = _v.reSize(this.getVectorSize());
        else if (this.getVectorSize() < _v.getVectorSize())
            myVector = this.reSize(_v.getVectorSize());

        double[] doubElementNew = new double[_v.getVectorSize()];
        for (int i = 0; i < _v.getVectorSize(); i++) {
            doubElementNew[i] = myVector.getElementatIndex(i) - _v.getElementatIndex(i);
        }
        return new Vector(doubElementNew);
    }

    public double dotProduct(Vector _v) {
        Vector myVector = this;
        if (this.getVectorSize() > _v.getVectorSize())
            _v = _v.reSize(this.getVectorSize());
        else if (this.getVectorSize() < _v.getVectorSize())
            myVector = this.reSize(_v.getVectorSize());

        double dotProduct = 0;
        for (int i = 0; i < _v.getVectorSize(); i++) {
            dotProduct += myVector.getElementatIndex(i) * _v.getElementatIndex(i);
        }
        return dotProduct;
    }

    public double cosineSimilarity(Vector _v) {
        Vector myVector = this;
        if (this.getVectorSize() > _v.getVectorSize())
            _v = _v.reSize(this.getVectorSize());
        else if (this.getVectorSize() < _v.getVectorSize())
            myVector = this.reSize(_v.getVectorSize());

        double doubDProduct = dotProduct(_v);
        double doubNorm1 = 0;
        double doubNorm2 = 0;
        for (int i = 0; i < _v.getVectorSize(); i++) {
            doubNorm1 += Math.pow(myVector.getElementatIndex(i), 2);
            doubNorm2 += Math.pow(_v.getElementatIndex(i), 2);
        }

        return doubDProduct / (Math.sqrt(doubNorm1) * Math.sqrt(doubNorm2));
    }

    @Override
    public boolean equals(Object _obj) {
        Vector v = (Vector) _obj;
        boolean boolEquals = true;

        if (this.getVectorSize()!= v.getVectorSize())
            return false;

        for (int i = 0; i < this.getVectorSize(); i++) {
            if (this.getElementatIndex(i) != v.getElementatIndex(i)) {
                boolEquals = false;
                break;
            }
        }
        return boolEquals;
    }

    @Override
    public String toString() {
        StringBuilder mySB = new StringBuilder();
        for (int i = 0; i < this.getVectorSize(); i++) {
            mySB.append(String.format("%.5f", doubElements[i])).append(",");
        }
        mySB.delete(mySB.length() - 1, mySB.length());
        return mySB.toString();
    }
}

