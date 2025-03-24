#ifndef __EXCEPTIONS__
#define __EXCEPTIONS__

#include <exception>
#include <string>


class WrongDimensionsException : std::exception {
    public:
        WrongDimensionsException(std::string message) { _message = message; }
        const char* what() const throw() { return _message.c_str(); }

    private:
        std::string _message;
};


class WrongSizeException : std::exception {
    public:
        WrongSizeException(std::string message) { _message = message; }
        const char* what() const throw() { return _message.c_str(); }
        
    private:
        std::string _message;
};


class NoTestDataInDataset : std::exception {
    public:
        NoTestDataInDataset(std::string message) { _message = message; }
        const char* what() const throw() { return _message.c_str(); }
        
    private:
        std::string _message;
};


class IndexOutOfRangeException : std::exception {
    public:
        IndexOutOfRangeException(std::string message) { _message = message; }
        const char* what() const throw() { return _message.c_str(); }
        
    private:
        std::string _message;
};


class NoGraphAttachedException : std::exception {
    public:
        NoGraphAttachedException(std::string message) { _message = message; }
        const char* what() const throw() { return _message.c_str(); }
    
    private:
        std::string _message;    
};


class MustBeScalarException : std::exception {
    public:
        MustBeScalarException(std::string message) { _message = message; }
        const char* what() const throw() { return _message.c_str(); }
    
    private:
        std::string _message;    
};


#endif