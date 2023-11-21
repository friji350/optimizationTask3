#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <cstdio>
#include <bits/stdc++.h>

using namespace std;

class ColumnVector {
public:
    vector<double> columnValues;
    unsigned long long columnSize;

    ColumnVector() {
        columnValues = vector<double>();
        columnSize = 0;
    }

    ColumnVector(ColumnVector const &C) {
        columnValues = C.columnValues;
        columnSize = C.columnSize;
    }

    ColumnVector(ColumnVector &C) {
        columnSize = C.columnSize;
        columnValues = C.columnValues;
    }

    ColumnVector(vector<double> &columnValues) {
        this->columnValues = columnValues;
        columnSize = columnValues.size();
    }

    ColumnVector(unsigned long long size) {
        columnSize = size;
        columnValues = vector<double>(size);
    }

    ColumnVector operator+(ColumnVector &b) {
        ColumnVector result(columnSize);
        for (int i = 0; i < columnSize; i++) {
            result.columnValues[i] = columnValues[i] + b.columnValues[i];
        }
        return result;
    }

    ColumnVector operator-(ColumnVector &b) {
        ColumnVector result(columnSize);
        for (int i = 0; i < columnSize; i++) {
            result.columnValues[i] = columnValues[i] - b.columnValues[i];
        }
        return result;
    }

    ColumnVector operator*(double scalar) {
        ColumnVector result(columnSize);
        for (int i = 0; i < columnSize; i++) {
            result.columnValues[i] = columnValues[i] * scalar;
        }
        return result;
    }

    ColumnVector operator/(double scalar) {
        ColumnVector result(columnSize);
        for (int i = 0; i < columnSize; i++) {
            result.columnValues[i] = columnValues[i] / scalar;
        }
        return result;
    }

    friend istream &operator>>(istream &stream, ColumnVector &columnVector) {
        for (int i = 0; i < columnVector.columnSize; i++) {
            stream >> columnVector.columnValues[i];
        }
        return stream;
    }

    friend ostream &operator<<(ostream &stream, ColumnVector &columnVector) {
        for (int i = 0; i < columnVector.columnSize; i++) {
            stream << columnVector.columnValues[i] << endl;
        }
        return stream;
    }

    double norm() {
        double result = 0;
        for (int i = 0; i < columnSize; i++) {
            result += (columnValues[i] * columnValues[i]);
        }
        return sqrt(result);
    }

    double &getElement(int j) {
        return this->columnValues[j];
    }

    void setElement(int j, double value) {
        this->columnValues[j] = value;
    }

    bool zeroCheck() {
        for (int i = 0; i < columnSize; i++) {
            if (columnValues[i] != 0) {
                return false;
            }
        }
        return true;
    }
};

class Matrix {
protected:
    vector<vector<double>> matrix;

    void MakeRectangle() {
        int maxSize = 0;
        for (auto &row: matrix) {
            if (row.size() > maxSize) {
                maxSize = row.size();
            }
        }
        for (auto &row: matrix) {
            row.resize(maxSize);
        }
    }

public:

    Matrix(int rows, int columns) {
        matrix.resize(rows);
        for (auto &row: matrix) {
            row.resize(columns);
        }
    }

    Matrix(vector<vector<double>> &m) : matrix(m) {
        MakeRectangle();
    }

    Matrix() = default;

    int Rows() {
        return matrix.size();
    }

    int Columns() {
        if (matrix.empty()) {
            return 0;
        }
        return matrix[0].size();
    }

    vector<double> &operator[](int i) {
        return matrix[i];
    }

    Matrix operator+(Matrix &matrixB) {
        int rows = Rows();
        int columns = Columns();
        if (rows != matrixB.Rows() || columns != matrixB.Columns()) {
            cout << ("Error: the dimensional problem occurred") << endl;
            return matrixB;
        }
        Matrix matrixD(rows, columns);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                matrixD[i][j] = matrix[i][j] + matrixB[i][j];
            }
        }
        return matrixD;
    }

    Matrix operator-(Matrix &matrixB) {
        int rows = Rows();
        int columns = Columns();
        if (rows != matrixB.Rows() || columns != matrixB.Columns()) {
            cout << ("Error: the dimensional problem occurred") << endl;
            return matrixB;
        }
        Matrix matrixE(rows, columns);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                matrixE[i][j] = matrix[i][j] - matrixB[i][j];
            }
        }
        return matrixE;
    }

    friend istream &operator>>(istream &in, Matrix &matrix) {
        for (int i = 0; i < matrix.Rows(); i++) {
            for (int j = 0; j < matrix.Columns(); j++) {
                in >> matrix[i][j];
            }
        }
        return in;
    }

    friend ostream &operator<<(ostream &out, Matrix &matrix) {
        for (int i = 0; i < matrix.Rows(); i++) {
            for (int j = 0; j < matrix.Columns(); j++) {
                out << setprecision(4) << fixed << matrix[i][j] << ' ';
            }
            cout << endl;
        }
        return out;
    }


    Matrix &operator=(Matrix matrix1) {
        matrix.resize(matrix1.Rows());
        for (auto &row: matrix) {
            row.resize(matrix1.Columns());
        }
        for (int i = 0; i < matrix1.Rows(); i++) {
            for (int j = 0; j < matrix1.Columns(); j++) {
                matrix[i][j] = matrix1[i][j];
            }
        }
        return *this;
    }


    static Matrix transpose(Matrix matrixA) {
        Matrix matrixG(matrixA.Columns(), matrixA.Rows());
        for (int i = 0; i < matrixA.Rows(); ++i) {
            for (int j = 0; j < matrixA.Columns(); ++j) {
                matrixG[j][i] = matrixA[i][j];
            }
        }
        return matrixG;
    }

    Matrix operator*(Matrix &matrixA) {
        int rows = Rows();
        int columns = Columns();
        Matrix matrixG(matrixA.Columns(), matrixA.Rows());
        if (columns != matrixA.Rows()) {
            matrixG = transpose(matrixA);
            if (columns != matrixG.Rows()) {
                cout << ("Error: the dimensional problem occurred") << endl;
                return matrixA;
            }
        } else {
            matrixG = matrixA;
        }
        Matrix matrixF(rows, matrixG.Columns());
        for (int i = 0; i < rows; ++i) {
            double count = 0;
            int i1 = 0;
            int matrixFcount = 0;
            for (int j = 0; j < matrixG.Rows(); ++j) {
                count += matrix[i][j] * matrixG[j][i1];
                if (j == matrixG.Rows() - 1) {
                    matrixF[i][matrixFcount] = count;
                    matrixFcount++;
                    if ((i1 + 1) <= matrixG.Columns() - 1) {
                        j = -1;
                        i1++;
                        count = 0;
                    }
                }
            }
        }
        return matrixF;
    }

    double findMaxDifference(int type, int index, int rowSize, int columnSize) {
        double minFirst = DBL_MAX;
        int minFirstIndex = -1;
        double minSecond = DBL_MAX;
        if (type == 0) {
            for (int i = 0; i < rowSize; ++i) {
                if (matrix[i][index] < minFirst && matrix[i][index] != -1) {
                    minFirst = matrix[i][index];
                    minFirstIndex = i;
                }
            }
            for (int i = 0; i < rowSize; ++i) {
                if (matrix[i][index] < minSecond && i != minFirstIndex && matrix[i][index] != -1) {
                    minSecond = matrix[i][index];
                }
            }
        } else {
            for (int i = 0; i < columnSize; ++i) {
                if (matrix[index][i] < minFirst && matrix[index][i] != -1) {
                    minFirst = matrix[index][i];
                    minFirstIndex = i;
                }
            }
            for (int i = 0; i < columnSize; ++i) {
                if (matrix[index][i] < minSecond && i != minFirstIndex && matrix[index][i] != -1) {
                    minSecond = matrix[index][i];
                }
            }
        }
        if (minFirst == DBL_MAX || minSecond == DBL_MAX) {
            return DBL_MAX;
        }
        return minSecond - minFirst;
    }

    bool oneElement(int rowSize, int columnSize) {
        int normalNum = 0;
        for (int i = 0; i < rowSize; i++) {
            for (int j = 0; j < columnSize; j++) {
                if (matrix[i][j] != -1) {
                    normalNum += 1;
                }
            }
        }
        if (normalNum <= 1) {
            return true;
        }
        return false;
    }
};


class SquareMatrix : public Matrix {
protected:
    int size;
public:
    SquareMatrix() = default;

    SquareMatrix(int size) : Matrix(size, size) {
        this->size = size;
        this->matrix.resize(size);
        for (auto &row: matrix) {
            row.resize(size);
        }

    }

    int Size() {
        return matrix.size();
    }

    SquareMatrix operator+(SquareMatrix &A) {
        auto B = *(Matrix *) &A;
        auto C = *(Matrix *) this + B;
        return *(SquareMatrix *) &C;
    }

    SquareMatrix operator-(SquareMatrix &A) {
        auto B = *(Matrix *) &A;
        auto C = *(Matrix *) this - B;
        return *(SquareMatrix *) &C;
    }

    SquareMatrix operator*(SquareMatrix &A) {
        auto B = *(Matrix *) &A;
        auto C = *(Matrix *) this * B;
        return *(SquareMatrix *) &C;
    }

    SquareMatrix transpose(SquareMatrix matrix) {
        auto B = *(Matrix *) &matrix;

        auto D = Matrix::transpose(B);
        return *(SquareMatrix *) &D;
    }
};


class IdentityMatrix : public SquareMatrix {

public:
    IdentityMatrix(int size) : SquareMatrix(size) {}

    IdentityMatrix fillIdentityMatrix() {
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; ++j) {
                if (i == j) {
                    this->matrix[i][j] = 1.00;
                } else {
                    this->matrix[i][j] = 0.00;
                }
            }
        }
        return *this;
    }
};

class PermutationMatrix : public SquareMatrix {
private:
    int size;
    int row1;
    int row2;
public:
    PermutationMatrix(int size, int row1, int row2) : SquareMatrix(size) {
        this->size = size;
        this->row1 = row1;
        this->row2 = row2;
        this->matrix.resize(size);
        for (auto &row: this->matrix) {
            row.resize(size);
        }
    }

    PermutationMatrix fillPermutationMatrix() {
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                this->matrix[i][j] = 0;
            }
        }
        for (int i = 0; i < size; i++) {
            if (i != row1 && i != row2) {
                this->matrix[i][i] = 1;
            }
        }
        this->matrix[row1][row2] = 1;
        this->matrix[row2][row1] = 1;
        return *this;
    }
};

class EliminationMatrix : public IdentityMatrix {
private:
    int size;
    int row;
    int column;
public:

    EliminationMatrix(int size, int row, int column)
            : IdentityMatrix(size) {
        this->size = size;
        this->row = row;
        this->column = column;
        this->matrix.resize(size);
        for (auto &rows: this->matrix) {
            rows.resize(size);
        }
    }

    Matrix fillEliminationMatrix(Matrix matrixA, int rowOfPivot, int columnOfPivot) {
        fillIdentityMatrix();
        this->matrix[row][column] = -1.00 * (double) (matrixA[row][column] / matrixA[rowOfPivot][columnOfPivot]);
        return *this;
    }
};

class REFMatrix : public SquareMatrix {
private:

    Matrix matrixA;
public:
    REFMatrix(int size, Matrix &matrixA) : SquareMatrix(size) { this->matrixA = matrixA; }

    int findDeterminant() {
        double det = 1;
        for (int i = 0; i < size; i++) {
            det *= matrixA[i][i];
        }
        return det;
    }
};

class InverseMatrix : public Matrix {
private:
    Matrix matrixA;
    SquareMatrix identityMatrix;
    int row;
    int column;
public:
    InverseMatrix(int row, int column, Matrix &matrixA) : Matrix(row, column) {
        this->matrixA = matrixA;
        this->row = row;
        this->column = column;
        this->identityMatrix = SquareMatrix(row);
    }

    void fillIdentityMatrix() {
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < row; ++j) {
                if (i == j) {
                    this->identityMatrix[i][j] = 1.00;
                } else {
                    this->identityMatrix[i][j] = 0.00;
                }
            }
        }
    }


    int numberOfstep = 0;

//    void printMatrices() {
//        for (int i = 0; i < row; ++i) {
//            for (int j = 0; j < column; ++j) {
//                cout << setprecision(2) << fixed << matrixA[i][j] << " ";
//            }
//            for (int j = 0; j < size; ++j) {
//                cout << setprecision(2) << fixed << identityMatrix[i][j] << " ";
//            }
//            cout << endl;
//        }
//    }
    void AugmentedMatrix() {
        fillIdentityMatrix();
        numberOfstep++;
//        printMatrices();
    }

    bool stopEliminationDirect(int pivotRow, int pivotColumn) {
        for (int i = pivotRow + 1; i < row; i++) {
            if (matrixA[i][pivotColumn] != 0) {
                return false;
            }
        }
        return true;
    }

    bool stopEliminationBack(int pivotRow, int pivotColumn) {
        for (int i = 0; i < pivotRow; i++) {
            if (matrixA[i][pivotColumn] != 0) {
                return false;
            }
        }
        return true;
    }


    int findPivot(int startRow, int column) {
        int max_index = startRow;
        for (int i = startRow; i < row; i++) {
            if (abs(matrixA[i][column]) > abs(matrixA[max_index][column])) {
                max_index = i;
            }
        }
        return max_index;
    }

    void findInverse() {
        AugmentedMatrix();

        int column = 0;
        for (int startRow = 0; startRow < row - 1; startRow++) {
            column = startRow;
            int pivot = findPivot(startRow, column);
            if (pivot != startRow) {
//                cout << "step #" << numberOfstep << ": permutation" << endl;
                numberOfstep++;

                PermutationMatrix permutationMatrix = PermutationMatrix(row, startRow, pivot);
                permutationMatrix.fillPermutationMatrix();
                matrixA = *(Matrix *) &permutationMatrix * matrixA;
                identityMatrix = permutationMatrix * (identityMatrix);
//                printMatrices();
            }
            for (int row = startRow + 1; row < this->row; row++) {
                if (!stopEliminationDirect(startRow, column)) {

                    numberOfstep++;
                    EliminationMatrix eliminationMatrix = EliminationMatrix(this->row, row, column);
                    eliminationMatrix.fillEliminationMatrix(matrixA, startRow, column);
                    matrixA = *(Matrix *) &eliminationMatrix * matrixA;
                    identityMatrix = eliminationMatrix * identityMatrix;
//                    printMatrices();
                }
            }
        }

        for (int startRow = row - 1; startRow >= 0; startRow--) {
            column = startRow;
            if (!stopEliminationBack(startRow, column)) {
                for (int row = startRow - 1; row >= 0; row--) {
//                    cout << "step #" << numberOfstep << ": elimination" << endl;
                    numberOfstep++;
                    EliminationMatrix eliminationMatrix = EliminationMatrix(this->row, row, column);
                    eliminationMatrix.fillEliminationMatrix(matrixA, startRow, column);
                    matrixA = *(Matrix *) &eliminationMatrix * matrixA;
                    identityMatrix = eliminationMatrix * identityMatrix;
//                    printMatrices();
                }
            }
        }

    }

    Matrix diagonalNormalization() {
        findInverse();
        for (int i = 0; i < row; ++i) {
            for (int j = 0; j < column; ++j) {
                identityMatrix[i][j] = (double) (identityMatrix[i][j] / matrixA[i][i]);
            }
            matrixA[i][i] = 1;
        }
//        cout << "Diagonal normalization:" << endl;
//        cout << "result:" << endl;
//        cout << identityMatrix;
        return identityMatrix;
    }
};

class LeastSquare : public Matrix {
private:
    Matrix initMatrix;
    Matrix vectorB;
    int degree;
    int m;
public:
    LeastSquare(Matrix initMatrix, Matrix vectorB, int degree, int m) {
        this->initMatrix = initMatrix;
        this->degree = degree;
        this->vectorB = vectorB;
        this->m = m;
    }

    Matrix makeMatrixA(int degree, int size, Matrix init) {
        Matrix matrixA(size, degree + 1);
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < degree + 1; ++j) {
                matrixA[i][j] = pow(init[i][0], j);
            }
        }
        return matrixA;
    }
};

class Solution {
public:
    void northWest(Matrix mainMatrix, int column, int row) {
        int currentX = 0;
        int currentY = 0;
        double S = 0;
        while (mainMatrix[row][column] != 0) {
            cout << "\ninitial basic feasible solution: x" << currentX << currentY << " = "
                 << mainMatrix[currentY][currentX] << "\n";
            double minValue = min(mainMatrix[currentY][column], mainMatrix[row][currentX]);
            S += mainMatrix[currentY][currentX] * minValue;
            mainMatrix[currentY][column] -= minValue;
            mainMatrix[row][currentX] -= minValue;
            if (mainMatrix[currentY][column] == 0) {
                currentY++;
            }
            if (mainMatrix[row][currentX] == 0) {
                currentX++;
            }
            mainMatrix[row][column] -= minValue;
            cout << mainMatrix;
            cout << "\n-------------------------------\n";
        }
        cout << "min cost of path according to NorthWest corner method: " << S << "\n";
    }

    void russel(Matrix mainMatrix, int column, int row) {
        double S = 0;
        int currentX = 0;
        int currentY = 0;
        Matrix additionalMatrix(row, column);
        ColumnVector u(row);
        ColumnVector v(column);
        while (mainMatrix[row][column] != 0) {

            double maxColumn = 0;
            double maxRow = 0;
            for (int i = 0; i < row; i++) {
                for (int k = 0; k < column; k++) {
                    if (mainMatrix[i][k] > maxRow) {
                        maxRow = mainMatrix[i][k];
                    }
                }
                u.setElement(i, maxRow);
                maxRow = 0;
            }
            for (int i = 0; i < column; i++) {
                for (int k = 0; k < row; k++) {
                    if (mainMatrix[k][i] > maxColumn) {
                        maxColumn = mainMatrix[k][i];
                    }
                }
                v.setElement(i, maxColumn);
                maxColumn = 0;
            }

            for (int i = 0; i < row; i++) {
                for (int j = 0; j < column; j++) {
                    if (mainMatrix[i][j] != -1) {
                        additionalMatrix[i][j] = mainMatrix[i][j] - u.getElement(i) - v.getElement(j);
                    }
                }
            }
            cout << "The general matrix: \n";
            cout << mainMatrix;
            cout << "\nAdditional matrix:\n";
            cout << additionalMatrix;
            cout << "\n-------------------------------\n";
            double minValue = 10000;
            int coordMinX;
            int coordMinY;
            for (int i = 0; i < row; i++) {
                for (int j = 0; j < column; j++) {
                    if (additionalMatrix[i][j] < minValue) {
                        minValue = additionalMatrix[i][j];
                        coordMinX = i;
                        coordMinY = j;
                    }
                }
            }
            currentX = coordMinX;
            currentY = coordMinY;
            cout << "basic variable: x" << currentX << currentY << " = "
                 << mainMatrix[currentX][currentY] << "\n";
            double currentCost = min(mainMatrix[currentX][column], mainMatrix[row][currentY]);
            S += currentCost * mainMatrix[currentX][currentY];
            mainMatrix[currentX][column] -= currentCost;
            mainMatrix[row][currentY] -= currentCost;
            if (mainMatrix[currentX][column] == 0) {
                for (int i = 0; i < column; i++) {
                    mainMatrix[currentX][i] = -1;
                    additionalMatrix[currentX][i] = 100000;
                }
            }
            if (mainMatrix[row][currentY] == 0) {
                for (int i = 0; i < row; i++) {
                    mainMatrix[i][currentY] = -1;
                    additionalMatrix[i][currentY] = 100000;
                }
            }
            mainMatrix[row][column] -= currentCost;

        }
        cout << mainMatrix;
        cout << additionalMatrix;
        cout << "\nmin cost of path according to Russell's method: " << S << "\n";
    }

    void vogel(Matrix matrix, int rowSize, int columnSize, ColumnVector supply, ColumnVector demand) {
        double S = 0;
        int iterNumber = 0;
        Matrix solveMatrix(rowSize, columnSize);
        for (int i = 0; i < rowSize; i++) {
            for (int j = 0; j < columnSize; j++) {
                solveMatrix[i][j] = matrix[i][j];
            }
        }

        while (!supply.zeroCheck() && !demand.zeroCheck()) {
            iterNumber += 1;
            int minDifference = DBL_MAX;
            int feasibleSolRowIndex = -1;
            int feasibleSolColumnIndex = -1;
            double diff = -1;

            for (int i = 0; i < rowSize; i++) {
                diff = solveMatrix.findMaxDifference(1, i, rowSize, columnSize);
                if (diff < minDifference) {
                    minDifference = diff;
                    feasibleSolRowIndex = i;
                }
            }

            for (int i = 0; i < columnSize; i++) {
                diff = solveMatrix.findMaxDifference(0, i, rowSize, columnSize);
                if (diff < minDifference) {
                    minDifference = diff;
                    feasibleSolRowIndex = -1;
                    feasibleSolColumnIndex = i;
                }
            }

            minDifference = DBL_MAX;
            if (feasibleSolRowIndex == -1) {
                for (int i = 0; i < rowSize; i++) {
                    if (solveMatrix[i][feasibleSolColumnIndex] < minDifference &&
                        solveMatrix[i][feasibleSolColumnIndex] != -1) {
                        minDifference = solveMatrix[i][feasibleSolColumnIndex];
                        feasibleSolRowIndex = i;
                    }
                }
            } else {
                for (int i = 0; i < columnSize; i++) {
                    if (solveMatrix[feasibleSolRowIndex][i] < minDifference &&
                        solveMatrix[feasibleSolRowIndex][i] != -1) {
                        minDifference = solveMatrix[feasibleSolRowIndex][i];
                        feasibleSolColumnIndex = i;
                    }
                }
            }

            if (supply.getElement(feasibleSolRowIndex) < demand.getElement(feasibleSolColumnIndex)) {
                S += supply.getElement(feasibleSolRowIndex) * solveMatrix[feasibleSolRowIndex][feasibleSolColumnIndex];
                for (int i = 0; i < columnSize; ++i) {
                    solveMatrix[feasibleSolRowIndex][i] = -1;
                }
                demand.setElement(feasibleSolColumnIndex,
                                  demand.getElement(feasibleSolColumnIndex) - supply.getElement(feasibleSolRowIndex));
                supply.setElement(feasibleSolRowIndex, 0);
            } else {
                S += demand.getElement(feasibleSolColumnIndex) *
                     solveMatrix[feasibleSolRowIndex][feasibleSolColumnIndex];
                for (int i = 0; i < rowSize; ++i) {
                    solveMatrix[i][feasibleSolColumnIndex] = -1;
                }
                supply.setElement(feasibleSolRowIndex,
                                  supply.getElement(feasibleSolRowIndex) - demand.getElement(feasibleSolColumnIndex));
                demand.setElement(feasibleSolColumnIndex, 0);
            }

            cout << "-----------------------" << '\n';
            cout << '\n';
            cout << "A" << feasibleSolRowIndex + 1 << ' ';
            cout << "B" << feasibleSolColumnIndex + 1 << '\n';
            cout << "Matrix C: \n";
            cout << solveMatrix;
            cout << '\n';
            cout << "Supply vector: \n";
            cout << supply;
            cout << '\n';
            cout << "Demand vector: \n";
            cout << demand;
            cout << '\n';
            cout << "S: " << S << '\n';

            if (solveMatrix.oneElement(rowSize, columnSize)) {
                iterNumber += 1;
                for (int i = 0; i < rowSize; i++) {
                    for (int j = 0; j < columnSize; j++) {
                        if (solveMatrix[i][j] != -1) {
                            S += supply.getElement(i) * solveMatrix[i][j];
                            supply.setElement(i, 0);
                            demand.setElement(j, 0);
                            solveMatrix[i][j] = -1;

                            cout << '\n';
                            cout << "-----------------------" << '\n';
                            cout << '\n';
                            cout << "Iteration number: " << iterNumber << '\n';
                            cout << "A" << i + 1 << ' ';
                            cout << "B" << j + 1 << '\n';
                            cout << "Matrix C: \n";
                            cout << solveMatrix;
                            cout << '\n';
                            cout << "Supply vector: \n";
                            cout << supply;
                            cout << '\n';
                            cout << "Demand vector: \n";
                            cout << demand;
                            cout << '\n';
                            cout << "S: " << S << '\n';
                        }
                    }
                }
            }
        }
        cout << "\nmin cost of path according to Vogel`s method: " << S << "\n";
    }
};

int main() {
    int row = 0;
    int column = 0;
    int supplySum = 0;
    int demandSum = 0;

    cout << "Enter the number of rows\n";
    cin >> row;
    cout << "Enter the number of columns\n";
    cin >> column;

    cout << "Enter the supply vector\n";
    ColumnVector supply(row);
    cin >> supply;

    cout << "Enter the demand vector\n";
    ColumnVector demand(column);
    cin >> demand;

    cout << "Enter the C matrix\n";
    Matrix mainMatrix(row + 1, column + 1);
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < column; j++) {
            int x;
            cin >> x;
            mainMatrix[i][j] = x;
        }
    }

    for (int i = 0; i < column; i++) {
        demandSum += demand.getElement(i);
        mainMatrix[row][i] = demand.getElement(i);
    }

    for (int i = 0; i < row; i++) {
        supplySum += supply.getElement(i);
        mainMatrix[i][column] = supply.getElement(i);
    }

    if (supplySum > demandSum) {
        cout << "The problem is not balanced!\n";
        exit(0);
    } else if (supplySum < demandSum) {
        cout << "The method is not applicable!\n";
        exit(0);
    } else {
        mainMatrix[row][column] = supplySum;
    }

    cout << mainMatrix;

    cout << "\nSolution by NorthWest corner method\n";
    Solution solution = *new Solution();
    solution.northWest(mainMatrix, column, row);
    cout << "\nSolution by Russell's method\n";
    solution.russel(mainMatrix, column, row);
    cout << "\nSolution by Vogel`s method\n";
    solution.vogel(mainMatrix, row, column, supply, demand);
    return 0;
}

