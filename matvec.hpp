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


    // Create a fake vector for some part of a tensor
    Vector(std::vector<ComponentType>::const_iterator begin,
           std::vector<ComponentType>::const_iterator end,
           size_t &alignment, bool &transpose) : _tensor(std::nullopt) {
        _begin = begin;
        _end = end;
        _alignment = alignment;
        _transpose = transpose;
    };


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

    // Vector vector multiplication
    ComponentType operator*(const Vector<ComponentType> &vec) const {
        assert(_transpose == true && vec._transpose == false);
        ComponentType result = 0;
        for (size_t i = 0; i < this->size(); i++) {
            result += vec(i) * *(_begin + (i * _alignment));
        }
        return result;
    }

    std::vector<ComponentType>::iterator &getBegin() {
        return _begin;
    }

    [[nodiscard]] size_t getAlignment() const {
        return _alignment;
    }

private:
    std::optional<Tensor<ComponentType>> _tensor;
    std::vector<ComponentType>::iterator _begin;
    std::vector<ComponentType>::iterator _end;
    size_t _alignment = 1;
    bool _transpose = false;
};

template<typename ComponentType>
Vector<ComponentType>::Vector(size_t size) {
    _tensor = Tensor<ComponentType>({size});
    _begin = _tensor->getIterator(0);
    _end = _tensor->getIterator(size);
}

template<typename ComponentType>
Vector<ComponentType>::Vector(size_t size, const ComponentType &fillValue) {
    _tensor = Tensor<ComponentType>({size}, fillValue);
    _begin = _tensor->getIterator(0);
    _end = _tensor->getIterator(size);
}

template<typename ComponentType>
Vector<ComponentType>::Vector(const std::string &filename) {
    _tensor = readTensorFromFile<ComponentType>(filename);
    _begin = _tensor->getIterator(0);
    _end = _tensor->getIterator(_tensor->numElements());
}

template<typename ComponentType>
inline size_t Vector<ComponentType>::size() const {
    return std::distance(_begin, _end);
}

template<typename ComponentType>
inline const ComponentType &Vector<ComponentType>::operator()(size_t idx) const {
    return *(_begin + (idx * _alignment));
}

template<typename ComponentType>
inline ComponentType &Vector<ComponentType>::operator()(size_t idx) {
    return *(_begin + (idx * _alignment));
}

template<typename ComponentType>
Tensor<ComponentType> &Vector<ComponentType>::tensor() {
    assert(_tensor.has_value() == true);
    return _tensor.value();
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

    // Overload * operator for Matrix Vector multiplication
    Vector<ComponentType> operator*(const Vector<ComponentType> &vector) const;

    std::vector<ComponentType>::const_iterator getRowIterator(const size_t &row) const {
        return _begin + row * _cols;
    }

    std::vector<ComponentType>::const_iterator getColumnIterator(const size_t &col) const {
        return _begin + col;
    }

private:
    Tensor<ComponentType> _tensor;
    std::vector<ComponentType>::iterator _begin;
    size_t _rows, _cols;
};


template<typename ComponentType>
Matrix<ComponentType>::Matrix(size_t rows, size_t cols): _tensor({rows, cols}), _begin(_tensor.getIterator(0)),
                                                         _rows(rows), _cols(cols) {

}

template<typename ComponentType>
Matrix<ComponentType>::Matrix(size_t rows, size_t cols, const ComponentType &fillValue): _tensor({rows, cols},
                                                                                                 fillValue),
                                                                                         _begin(_tensor.getIterator(0)),
                                                                                         _rows(rows), _cols(cols) {

}

template<typename ComponentType>
Matrix<ComponentType>::Matrix(const std::string &filename) {
    _tensor = readTensorFromFile<ComponentType>(filename);
    _begin = _tensor.getIterator(0);
    _rows = _tensor.shape()[0];
    _cols = _tensor.shape()[1];
}

template<typename ComponentType>
size_t Matrix<ComponentType>::rows() const {
    return _rows;
}

template<typename ComponentType>
size_t Matrix<ComponentType>::cols() const {
    return _cols;
}

template<typename ComponentType>
const ComponentType &Matrix<ComponentType>::operator()(size_t row, size_t col) const {
    return _tensor({row, col});
}

template<typename ComponentType>
ComponentType &Matrix<ComponentType>::operator()(size_t row, size_t col) {
    return _tensor({row, col});
}

template<typename ComponentType>
Tensor<ComponentType> &Matrix<ComponentType>::tensor() {
    return _tensor;
}

template<typename ComponentType>
Vector<ComponentType> Matrix<ComponentType>::operator*(const Vector<ComponentType> &vec) const {
    assert(this->cols() == vec.size());
    Vector<ComponentType> result(vec.size());
    auto matIt = _begin;
    for (size_t row = 0; row < this->rows(); ++row) {
        ComponentType temp = 0;
        for (size_t col = 0; col < _cols; ++col) {
            temp += *matIt * vec(col);
            ++matIt;
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

