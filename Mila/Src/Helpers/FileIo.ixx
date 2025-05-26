module;  
#include <cstdio>  
#include <cstdlib>  
#include <iostream>  

export module Helpers.FileIo.Unused;  

namespace Mila::Helpers
{  
	// TODO: Remove.
    // TJT: This file is no longer needed as file I/O is handled by std::fstream.
	//      I will leave it here for reference. 
   /**
    * @brief Opens a file with error checking.
    * 
    * @param path The path to the file.
    * @param mode The mode in which to open the file.
    * @param file The source file name where the function is called.
    * @param line The line number in the source file where the function is called.
    * @return FILE* Pointer to the opened file.
    */
   extern inline FILE* fopen_check( const char* path, const char* mode, const char* file, int line ) {  
       FILE* fp = fopen( path, mode );  
       if ( fp == nullptr ) {  
           std::cerr << "Error: Failed to open file '" << path << "' at " << file << ":" << line << "\n";  
           std::cerr << "Error details:\n";  
           std::cerr << "  File: " << file << "\n";  
           std::cerr << "  Line: " << line << "\n";  
           std::cerr << "  Path: " << path << "\n";  
           std::cerr << "  Mode: " << mode << "\n";  
           std::cerr << "---> HINT 1: dataset files/code have moved to dev/data recently (May 20, 2024). You may have to mv them from the legacy data/ dir to dev/data/(dataset), or re-run the data preprocessing script. Refer back to the main README\n";  
           std::cerr << "---> HINT 2: possibly try to re-run `python train_gpt2.py`\n";  
           std::exit( EXIT_FAILURE );  
       }  
       return fp;  
   }  

   /**
    * @brief Opens a file with error checking.
    * 
    * @param path The path to the file.
    * @param mode The mode in which to open the file.
    * @return FILE* Pointer to the opened file.
    */
   export FILE* fopenCheck( const char* path, const char* mode ) {  
       return fopen_check( path, mode, __FILE__, __LINE__ );  
   }  

   /**
    * @brief Reads from a file with error checking.
    * 
    * @param ptr Pointer to the buffer where the read data will be stored.
    * @param size Size of each element to be read.
    * @param nmemb Number of elements to be read.
    * @param stream Pointer to the file stream.
    * @param file The source file name where the function is called.
    * @param line The line number in the source file where the function is called.
    */
   extern inline void fread_check( void* ptr, size_t size, size_t nmemb, FILE* stream, const char* file, int line ) {  
       size_t result = fread( ptr, size, nmemb, stream );  
       if ( result != nmemb ) {  
           if ( feof( stream ) ) {  
               std::cerr << "Error: Unexpected end of file at " << file << ":" << line << "\n";  
           }  
           else if ( ferror( stream ) ) {  
               std::cerr << "Error: File read error at " << file << ":" << line << "\n";  
           }  
           else {  
               std::cerr << "Error: Partial read at " << file << ":" << line << ". Expected " << nmemb << " elements, read " << result << "\n";  
           }  
           std::cerr << "Error details:\n";  
           std::cerr << "  File: " << file << "\n";  
           std::cerr << "  Line: " << line << "\n";  
           std::cerr << "  Expected elements: " << nmemb << "\n";  
           std::cerr << "  Read elements: " << result << "\n";  
           std::exit( EXIT_FAILURE );  
       }  
   }  

   /**
    * @brief Reads from a file with error checking.
    * 
    * @param ptr Pointer to the buffer where the read data will be stored.
    * @param size Size of each element to be read.
    * @param nmemb Number of elements to be read.
    * @param stream Pointer to the file stream.
    */
   export void freadCheck( void* ptr, size_t size, size_t nmemb, FILE* stream ) {  
       return fread_check( ptr, size, nmemb, stream, __FILE__, __LINE__ );  
   }  

   /**
    * @brief Closes a file with error checking.
    * 
    * @param fp Pointer to the file to be closed.
    * @param file The source file name where the function is called.
    * @param line The line number in the source file where the function is called.
    */
   extern inline void fclose_check( FILE* fp, const char* file, int line ) {  
       if ( fclose( fp ) != 0 ) {  
           std::cerr << "Error: Failed to close file at " << file << ":" << line << "\n";  
           std::cerr << "Error details:\n";  
           std::cerr << "  File: " << file << "\n";  
           std::cerr << "  Line: " << line << "\n";  
           std::exit( EXIT_FAILURE );  
       }  
   }  

   /**
    * @brief Closes a file with error checking.
    * 
    * @param fp Pointer to the file to be closed.
    */
   export void fcloseCheck( FILE* fp ) {  
       return fclose_check( fp, __FILE__, __LINE__ );  
   }  

   /**
    * @brief Seeks to a position in a file with error checking.
    * 
    * @param stream Pointer to the file stream.
    * @param offset Offset from the position specified by whence.
    * @param whence Position from where offset is added.
    * @param file The source file name where the function is called.
    * @param line The line number in the source file where the function is called.
    */
   extern inline void fseek_check( FILE* stream, long offset, int whence, const char* file, int line ) {  
       if ( fseek( stream, offset, whence ) != 0 ) {  
           std::cerr << "Error: Failed to seek in file at " << file << ":" << line << "\n";  
           std::cerr << "Error details:\n";  
           std::cerr << "  File: " << file << "\n";  
           std::cerr << "  Line: " << line << "\n";  
           std::cerr << "  Offset: " << offset << "\n";  
           std::cerr << "  Whence: " << whence << "\n";  
           std::exit( EXIT_FAILURE );  
       }  
   }  

   /**
    * @brief Seeks to a position in a file with error checking.
    * 
    * @param stream Pointer to the file stream.
    * @param offset Offset from the position specified by whence.
    * @param whence Position from where offset is added.
    */
   export void fseekCheck( FILE* stream, long offset, int whence ) {  
       return fseek_check( stream, offset, whence, __FILE__, __LINE__ );  
   }  
}