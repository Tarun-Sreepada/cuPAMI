#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <arrow/api.h>
#include <arrow/io/api.h>
#include <arrow/csv/api.h>
#include <parquet/arrow/writer.h>
#include <arrow/status.h>
#include <arrow/util/logging.h>

// Macro to throw an exception if an Arrow Status is not OK
#define THROW_IF_NOT_OK(status)                                           \
    do {                                                                 \
        if (!(status).ok()) {                                            \
            throw std::runtime_error((status).ToString());               \
        }                                                                \
    } while (0)

// Function to replace file extension
std::string ReplaceExtension(const std::string& filename, const std::string& new_ext) {
    size_t last_dot = filename.find_last_of(".");
    if (last_dot == std::string::npos) {
        return filename + new_ext; // No extension, just append the new one
    }
    return filename.substr(0, last_dot) + new_ext;
}

// Function to read a CSV file into an Arrow Table
std::shared_ptr<arrow::Table> ReadCSVToArrowTable(const std::string& csv_path) {
    // Open the CSV file
    std::ifstream infile(csv_path);
    if (!infile.is_open()) {
        throw std::runtime_error("Unable to open CSV file: " + csv_path);
    }

    // Read the entire CSV file into memory
    std::stringstream buffer;
    buffer << infile.rdbuf();
    infile.close();

    // Create a memory input stream from the CSV data
    auto input = std::make_shared<arrow::io::BufferReader>(
        arrow::Buffer::FromString(buffer.str()));

    // Configure CSV read options
    auto read_options = arrow::csv::ReadOptions::Defaults();
    auto parse_options = arrow::csv::ParseOptions::Defaults();
    auto convert_options = arrow::csv::ConvertOptions::Defaults();

    // Read the CSV file into an Arrow Table
    auto table_reader_result = arrow::csv::TableReader::Make(
        arrow::io::default_io_context(), input, read_options, parse_options, convert_options);

    if (!table_reader_result.ok()) {
        throw std::runtime_error("Failed to create TableReader: " + table_reader_result.status().ToString());
    }

    auto table_reader = table_reader_result.ValueOrDie();
    auto table_result = table_reader->Read();

    if (!table_result.ok()) {
        throw std::runtime_error("Failed to read CSV into Arrow Table: " + table_result.status().ToString());
    }

    return table_result.ValueOrDie();
}

// Function to write Arrow Table to Parquet
void WriteParquet(const std::shared_ptr<arrow::Table>& table, const std::string& parquet_path) {
    try {
        // Open a file output stream for writing Parquet file
        std::shared_ptr<arrow::io::FileOutputStream> outfile;
        auto result = arrow::io::FileOutputStream::Open(parquet_path);
        if (!result.ok()) {
            throw std::runtime_error("Failed to open file output stream: " + result.status().ToString());
        }
        outfile = *result;

        // Write the Arrow Table to Parquet
        THROW_IF_NOT_OK(parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), outfile, 1024));

        std::cout << "Converted to Parquet: " << parquet_path << std::endl;
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to write Parquet file: " + std::string(e.what()));
    }
}


int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <csv_file>" << std::endl;
        return EXIT_FAILURE;
    }

    std::string csv_file = argv[1];
    std::string parquet_file = ReplaceExtension(csv_file, ".parquet");

    try {
        auto table = ReadCSVToArrowTable(csv_file);
        WriteParquet(table, parquet_file);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
