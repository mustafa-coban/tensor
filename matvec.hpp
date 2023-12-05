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
    Vector(const std::string &filename);

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

private:
    Tensor<ComponentType> tensor_;
};

template<typename ComponentType>
Vector<ComponentType>::Vector(size_t size):tensor_({size}) {

}

template<typename ComponentType>
Vector<ComponentType>::Vector(size_t size, const ComponentType &fillValue):tensor_({size}, fillValue) {

}

template<typename ComponentType>
Vector<ComponentType>::Vector(const std::string &filename) {
    tensor_ = readTensorFromFile<ComponentType>(filename);
}

template<typename ComponentType>
size_t Vector<ComponentType>::size() const {
    return tensor_.numElements();
}

template<typename ComponentType>
const ComponentType &Vector<ComponentType>::operator()(size_t idx) const {
    return tensor_({idx});
}

template<typename ComponentType>
ComponentType &Vector<ComponentType>::operator()(size_t idx) {
    return tensor_({idx});
}

template<typename ComponentType>
Tensor<ComponentType> &Vector<ComponentType>::tensor() {
    return tensor_;
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
    Matrix(const std::string &filename);

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

    // Overload * operator for Matrix Vector multiplication
    Vector<ComponentType> operator* (const Vector<ComponentType>& vector) const;
private:
    Tensor<ComponentType> tensor_;
};


template<typename ComponentType>
Matrix<ComponentType>::Matrix(size_t rows, size_t cols): tensor_({rows, cols}) {

}

template<typename ComponentType>
Matrix<ComponentType>::Matrix(size_t rows, size_t cols, const ComponentType &fillValue): tensor_({rows, cols},
                                                                                                 fillValue) {

}

template<typename ComponentType>
Matrix<ComponentType>::Matrix(const std::string &filename) {
    tensor_ = readTensorFromFile<ComponentType>(filename);
}

template<typename ComponentType>
size_t Matrix<ComponentType>::rows() const {
    return tensor_.shape()[0];
}

template<typename ComponentType>
size_t Matrix<ComponentType>::cols() const {
    return tensor_.shape()[1];
}

template<typename ComponentType>
const ComponentType &Matrix<ComponentType>::operator()(size_t row, size_t col) const {
    return tensor_({row, col});
}

template<typename ComponentType>
ComponentType &Matrix<ComponentType>::operator()(size_t row, size_t col) {
    return tensor_({row, col});
}

template<typename ComponentType>
Tensor<ComponentType> &Matrix<ComponentType>::tensor() {
    return tensor_;
}

template<typename ComponentType>
Vector<ComponentType> Matrix<ComponentType>::operator*(const Vector<ComponentType> &vec) const{
    assert(this->cols() == vec.size());
    Vector<ComponentType> result(vec.size());
    for (size_t row = 0; row < this->rows(); ++row) {
        ComponentType temp = 0;
        for (size_t col = 0; col < vec.size(); ++col) {
            temp += (*this)(row, col) * vec(col);
        }
        result(row) = temp;
    }
    return result;
}

// Performs a matrix-vector multiplication.
template<typename ComponentType>
Vector<ComponentType> matvec(const Matrix<ComponentType> &mat, const Vector<ComponentType> &vec) {
    return mat * vec;
}

