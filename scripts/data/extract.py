import csv
import sys
import re
from pathlib import Path

def remove_tags(text):
    """Removes text enclosed in angle brackets <>."""
    return text
    # return re.sub(r'<.*?>', '', text).strip()

def process_directory_recursively_to_tsv(input_directory, output_filepath, output_encoding='utf-8'):
    """
    Recursively processes all .txt files in a directory and its subdirectories,
    attempting to detect input encoding (UTF-8 or CP1252), combines multi-line
    sentence fragments, removes <tags>, and writes the combined sentences to a
    single TSV file with the specified output encoding.

    Args:
        input_directory (str): Path to the root directory to search for .txt files.
        output_filepath (str): Path for the single output .tsv file.
        output_encoding (str): The encoding for writing the output file. Defaults to 'utf-8'.
    """
    print(f"Starting recursive processing with encoding detection...")
    print(f"Input directory: {input_directory}")
    print(f"Output file: {output_filepath} (TSV format)")
    # Input encoding is now detected per file
    print(f"Using output encoding: {output_encoding}")
    print("Combining multi-line entries and removing <tags> from all .txt files found recursively.")

    rows_written = 0
    total_lines_processed = 0
    files_processed_count = 0

    # State variables - maintained across all files processed
    current_identifier = None
    accumulated_mapudungun = ""
    accumulated_spanish = ""

    # Encodings to try, in order of preference
    # 'utf-8-sig' handles BOM, 'utf-8' is standard, 'cp1252' is fallback
    encodings_to_try = ['utf-8-sig', 'utf-8', 'cp1252']

    dir_path = Path(input_directory)
    if not dir_path.is_dir():
        print(f"\nERROR: Input directory not found or is not a directory: '{input_directory}'", file=sys.stderr)
        return

    try:
        txt_files = sorted(list(dir_path.rglob('*.txt')))
        if not txt_files:
            print(f"\nWARNING: No '.txt' files found in '{input_directory}' or its subdirectories.", file=sys.stderr)
            try:
                with open(output_filepath, 'w', newline='', encoding=output_encoding) as outfile:
                    tsv_writer = csv.writer(outfile, delimiter='\t')
                    tsv_writer.writerow(['Identifier', 'Mapudungun', 'Spanish'])
                print("\nCreated empty output file with headers.")
                return
            except IOError as e:
                 print(f"\nERROR: Could not open output file '{output_filepath}' for writing headers. {e}", file=sys.stderr)
                 return

        print(f"Found {len(txt_files)} '.txt' files to process.")

        with open(output_filepath, 'w', newline='', encoding=output_encoding) as outfile:
            tsv_writer = csv.writer(outfile, delimiter='\t')
            tsv_writer.writerow(['Identifier', 'Mapudungun', 'Spanish']) # Write header once

            for txt_filepath in txt_files:
                relative_path = txt_filepath.relative_to(dir_path)
                print(f"--- Processing file: {relative_path} ---")
                files_processed_count += 1

                # --- Encoding Detection ---
                decoded_lines = None
                detected_encoding = None
                file_content_to_process = []

                for encoding_attempt in encodings_to_try:
                    try:
                        print(f"    Trying encoding: {encoding_attempt}...")
                        # Read the entire file to validate encoding
                        with open(txt_filepath, 'r', encoding=encoding_attempt) as infile:
                            file_content_to_process = infile.readlines()
                        # If reading succeeded without error:
                        detected_encoding = encoding_attempt
                        print(f"    Successfully decoded using: {detected_encoding}")
                        break # Stop trying other encodings for this file
                    except UnicodeDecodeError:
                        print(f"    Failed decoding as {encoding_attempt}.")
                        # Continue to the next encoding in the list
                    except (IOError, OSError) as e: # Catch file read errors early
                        print(f"\nERROR: Could not read file '{relative_path}' due to OS/IO error. Skipping.", file=sys.stderr)
                        print(f"Error details: {e}", file=sys.stderr)
                        file_content_to_process = None # Ensure we skip processing
                        break # Stop trying encodings for this file
                    except Exception as e: # Catch other unexpected errors during read attempt
                        print(f"\nERROR: Unexpected error reading file '{relative_path}' with encoding {encoding_attempt}. Skipping.", file=sys.stderr)
                        print(f"Error details: {e}", file=sys.stderr)
                        file_content_to_process = None
                        break # Stop trying encodings for this file

                # --- Check if decoding failed for all attempts ---
                if detected_encoding is None or file_content_to_process is None:
                    print(f"\nERROR: Could not decode file '{relative_path}' using any tried encoding ({encodings_to_try}). Skipping.", file=sys.stderr)
                    current_identifier = None # Reset state if file fails
                    accumulated_mapudungun = ""
                    accumulated_spanish = ""
                    continue # Skip to the next file

                # --- Process the decoded lines ---
                file_lines_processed = 0
                try:
                    for line in file_content_to_process:
                        total_lines_processed += 1
                        file_lines_processed += 1
                        cleaned_line = line.strip()

                        if not cleaned_line or cleaned_line.startswith(';'):
                            continue

                        if cleaned_line.endswith(':') and not cleaned_line.startswith(('M:', 'C:')):
                            if current_identifier is not None and (accumulated_mapudungun or accumulated_spanish):
                                tsv_writer.writerow([
                                    current_identifier,
                                    accumulated_mapudungun.strip(),
                                    accumulated_spanish.strip()
                                ])
                                rows_written += 1
                            current_identifier = cleaned_line[:-1]
                            accumulated_mapudungun = ""
                            accumulated_spanish = ""
                        elif cleaned_line.startswith("M:"):
                            fragment = cleaned_line[2:].strip()
                            cleaned_fragment = remove_tags(fragment)
                            accumulated_mapudungun += cleaned_fragment + " "
                        elif cleaned_line.startswith("C:"):
                            fragment = cleaned_line[2:].strip()
                            cleaned_fragment = remove_tags(fragment)
                            accumulated_spanish += cleaned_fragment + " "

                    print(f"--- Finished processing file: {relative_path} ({file_lines_processed} lines) ---")

                except UnicodeEncodeError as e:
                     # This error happens during *writing* if output encoding fails
                    print(f"\nERROR: Encoding issue writing output for identifier '{current_identifier}' from file '{relative_path}' near line {file_lines_processed}.", file=sys.stderr)
                    print(f"Output encoding '{output_encoding}' failed. Details: {e}", file=sys.stderr)
                    raise # Stop process if writing fails
                except Exception as e:
                    # Catch errors during the processing of lines *after* successful decoding
                    print(f"\nERROR: Unexpected error processing decoded content from '{relative_path}' near line {file_lines_processed}. Skipping rest of file.", file=sys.stderr)
                    print(f"Error details: {e}", file=sys.stderr)
                    import traceback
                    traceback.print_exc(file=sys.stderr)
                    current_identifier = None # Reset state
                    accumulated_mapudungun = ""
                    accumulated_spanish = ""
                    continue # Skip to next file


            # --- After processing ALL files ---
            if current_identifier is not None and (accumulated_mapudungun or accumulated_spanish):
                 try:
                     tsv_writer.writerow([
                         current_identifier,
                         accumulated_mapudungun.strip(),
                         accumulated_spanish.strip()
                     ])
                     rows_written += 1
                 except UnicodeEncodeError as e:
                     print(f"\nERROR: Encoding issue writing final output for identifier '{current_identifier}'. {e}", file=sys.stderr)
                     raise

        print(f"\nProcessing complete!")
        print(f"Processed {files_processed_count} '.txt' files ({total_lines_processed} lines total).")
        print(f"Wrote {rows_written} combined rows to {output_filepath}")

    except IOError as e:
        print(f"\nERROR: Could not open output file '{output_filepath}' or access input directory '{input_directory}'. {e}", file=sys.stderr)
    except LookupError as e:
        print(f"\nERROR: Invalid encoding name specified (likely output encoding). {e}", file=sys.stderr)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)

# --- How to use the script ---

input_dir = 'Mapudungun Data/AUDIO'
output_file = 'combined_output.tsv'

# 3. Define the encodings.
output_encoding_type = 'utf-8'

# Run the processing function
process_directory_recursively_to_tsv(input_dir,
                         output_file,
                         output_encoding=output_encoding_type)
