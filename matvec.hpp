#pragma once

#include "tensor.hpp"

template<typename ComponentType>
class Vector {
public:
    // Default-constructor.
    Vector() = default;

    // Constructor for vector of certain size.
    explicit Vector(size_t size);

    // Constructor for vector of certain size with constant fill-value.
    Vector(size_t size, const ComponentType &fillValue);

    // Constructing vector from file.
    explicit Vector(const std::string &filename);

    // Number of elements in this vector.
    [[nodiscard]] size_t size() const;

    // Element access function
    const ComponentType &
    operator()(size_t idx) const;

    // Element mutation function
    ComponentType &
    operator()(size_t idx);

    // Reference to internal tensor.
    Tensor<ComponentType> &tensor();

    // Reference to internal tensor.
    const Tensor<ComponentType> &tensor() const;

    // Vector vector multiplication
    ComponentType operator*(const Vector<ComponentType> &vec) const;

private:
    Tensor<ComponentType> _tensor;
    bool _transpose = false;
};

template<typename ComponentType>
Vector<ComponentType>::Vector(size_t size) {
    _tensor = Tensor<ComponentType>({size});
}

template<typename ComponentType>
Vector<ComponentType>::Vector(size_t size, const ComponentType &fillValue) {
    _tensor = Tensor<ComponentType>({size}, fillValue);
}

template<typename ComponentType>
Vector<ComponentType>::Vector(const std::string &filename) {
    _tensor = readTensorFromFile<ComponentType>(filename);
}

template<typename ComponentType>
inline size_t Vector<ComponentType>::size() const {
    return _tensor.numElements();
}

template<typename ComponentType>
inline const ComponentType &Vector<ComponentType>::operator()(size_t idx) const {
    return _tensor({idx});
}

template<typename ComponentType>
inline ComponentType &Vector<ComponentType>::operator()(size_t idx) {
    return _tensor({idx});
}

template<typename ComponentType>
inline Tensor<ComponentType> &Vector<ComponentType>::tensor() {
    return _tensor;
}

template<typename ComponentType>
inline const Tensor<ComponentType> &Vector<ComponentType>::tensor() const {
    return _tensor;
}

template<typename ComponentType>
inline ComponentType Vector<ComponentType>::operator*(const Vector<ComponentType> &vec) const {
    assert(_transpose == true && vec._transpose == false);
    return std::inner_product(_tensor->begin({}), _tensor.end({}), vec._tensor->begin({}), 0);
}

template<typename ComponentType>
class Matrix {
public:
    // Default-constructor.
    Matrix() = default;

    // Constructor for matrix of certain size.
    explicit Matrix(size_t rows, size_t cols);

    // Constructor for matrix of certain size with constant fill-value.
    Matrix(size_t rows, size_t cols, const ComponentType &fillValue);

    // Constructing matrix from file.
    explicit Matrix(const std::string &filename);

    // Number of rows.
    [[nodiscard]] size_t rows() const;

    // Number of columns.
    [[nodiscard]] size_t cols() const;

    // Element access function
    const ComponentType &
    operator()(size_t row, size_t col) const;

    // Element mutation function
    ComponentType &
    operator()(size_t row, size_t col);

    // Reference to internal tensor.
    Tensor<ComponentType> &tensor();

    // Reference to internal tensor.
    const Tensor<ComponentType> &tensor() const;

    // Overload * operator for Matrix Vector multiplication
    Vector<ComponentType> operator*(const Vector<ComponentType> &vector) const;


private:
    Tensor<ComponentType> _tensor;
};


template<typename ComponentType>
Matrix<ComponentType>::Matrix(size_t rows, size_t cols): _tensor({rows, cols}) {

}

template<typename ComponentType>
Matrix<ComponentType>::Matrix(size_t rows, size_t cols, const ComponentType &fillValue): _tensor({rows, cols},
                                                                                                 fillValue) {

}

template<typename ComponentType>
Matrix<ComponentType>::Matrix(const std::string &filename) {
    _tensor = readTensorFromFile<ComponentType>(filename);
}

template<typename ComponentType>
inline size_t Matrix<ComponentType>::rows() const {
    return _tensor.shape()[0];
}

template<typename ComponentType>
inline size_t Matrix<ComponentType>::cols() const {
    return _tensor.shape()[1];
}

template<typename ComponentType>
inline const ComponentType &Matrix<ComponentType>::operator()(size_t row, size_t col) const {
    return _tensor({row, col});
}

template<typename ComponentType>
inline ComponentType &Matrix<ComponentType>::operator()(size_t row, size_t col) {
    return _tensor({row, col});
}

template<typename ComponentType>
inline Tensor<ComponentType> &Matrix<ComponentType>::tensor() {
    return _tensor;
}

template<typename ComponentType>
inline const Tensor<ComponentType> &Matrix<ComponentType>::tensor() const {
    return _tensor;
}

template<typename ComponentType>
inline Vector<ComponentType> Matrix<ComponentType>::operator*(const Vector<ComponentType> &vec) const {
    assert(this->cols() == vec.size());
    Vector<ComponentType> result(vec.size());

    int rows = static_cast<int>(this->rows());
    for (int row = 0; row < rows; ++row) {
        result(row) = std::inner_product(_tensor.begin({row, -1}), _tensor.end({row, -1}), vec.tensor().begin({}), 0);
    }
    return result;
}

// Performs a matrix-vector multiplication.
template<typename ComponentType>
inline Vector<ComponentType> matvec(const Matrix<ComponentType> &mat, const Vector<ComponentType> &vec) {
    return mat * vec;
}

