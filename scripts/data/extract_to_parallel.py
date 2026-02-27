import sys
import re # re is not used by remove_tags as per your provided code, but kept if other logic might need it
from pathlib import Path

def remove_tags(text):
    """Removes text enclosed in angle brackets <>.
    As per your provided code, this function currently returns text as is."""
    return text
    # return re.sub(r'<.*?>', '', text).strip() # Original tag removal line

def process_directory_to_parallel_texts(input_directory, output_basename, output_encoding='utf-8'):
    """
    Recursively processes all .txt files in a directory and its subdirectories,
    attempting to detect input encoding (UTF-8 or CP1252), combines multi-line
    sentence fragments, and writes the combined sentences to two parallel text files
    (one for Mapudungun, one for Spanish) with the specified output encoding.

    Args:
        input_directory (str): Path to the root directory to search for .txt files.
        output_basename (str): The base name for the output parallel text files.
                               E.g., if 'parallel_text', files will be 'parallel_text.arn.txt'
                               and 'parallel_text.es.txt'.
        output_encoding (str): The encoding for writing the output files. Defaults to 'utf-8'.
    """
    mapudungun_ext = ".arn" # Or simply .arn
    spanish_ext = ".es"    # Or simply .es

    mapudungun_output_filepath = Path(output_basename + mapudungun_ext)
    spanish_output_filepath = Path(output_basename + spanish_ext)

    print(f"Starting recursive processing for parallel text files...")
    print(f"Input directory: {input_directory}")
    print(f"Output Mapudungun file: {mapudungun_output_filepath}")
    print(f"Output Spanish file: {spanish_output_filepath}")
    print(f"Using output encoding: {output_encoding}")
    print("Combining multi-line entries from all .txt files found recursively.")

    lines_written = 0 # Changed from rows_written to lines_written
    total_lines_processed = 0
    files_processed_count = 0

    # State variables - maintained across all files processed
    current_identifier = None
    accumulated_mapudungun = ""
    accumulated_spanish = ""

    # Encodings to try
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
                # Create empty output files if no input
                with open(mapudungun_output_filepath, 'w', encoding=output_encoding) as map_outfile, \
                     open(spanish_output_filepath, 'w', encoding=output_encoding) as es_outfile:
                    pass # Just create them
                print("\nCreated empty output files.")
                return
            except IOError as e:
                 print(f"\nERROR: Could not create empty output files. {e}", file=sys.stderr)
                 return

        print(f"Found {len(txt_files)} '.txt' files to process.")

        # Open both output files for writing
        with open(mapudungun_output_filepath, 'w', encoding=output_encoding) as map_outfile, \
             open(spanish_output_filepath, 'w', encoding=output_encoding) as es_outfile:

            for txt_filepath in txt_files:
                relative_path = txt_filepath.relative_to(dir_path)
                print(f"--- Processing file: {relative_path} ---")
                files_processed_count += 1

                # --- Encoding Detection ---
                detected_encoding = None
                file_content_to_process = []

                for encoding_attempt in encodings_to_try:
                    try:
                        print(f"    Trying encoding: {encoding_attempt}...")
                        with open(txt_filepath, 'r', encoding=encoding_attempt) as infile:
                            file_content_to_process = infile.readlines()
                        detected_encoding = encoding_attempt
                        print(f"    Successfully decoded using: {detected_encoding}")
                        break
                    except UnicodeDecodeError:
                        print(f"    Failed decoding as {encoding_attempt}.")
                    except (IOError, OSError) as e:
                        print(f"\nERROR: Could not read file '{relative_path}' due to OS/IO error. Skipping.", file=sys.stderr)
                        print(f"Error details: {e}", file=sys.stderr)
                        file_content_to_process = None
                        break
                    except Exception as e:
                        print(f"\nERROR: Unexpected error reading file '{relative_path}' with encoding {encoding_attempt}. Skipping.", file=sys.stderr)
                        print(f"Error details: {e}", file=sys.stderr)
                        file_content_to_process = None
                        break

                if detected_encoding is None or file_content_to_process is None:
                    print(f"\nERROR: Could not decode file '{relative_path}' using any tried encoding ({encodings_to_try}). Skipping.", file=sys.stderr)
                    current_identifier = None
                    accumulated_mapudungun = ""
                    accumulated_spanish = ""
                    continue

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
                                # Write to parallel files
                                map_outfile.write(accumulated_mapudungun.strip() + "\n")
                                es_outfile.write(accumulated_spanish.strip() + "\n")
                                lines_written += 1
                            current_identifier = cleaned_line[:-1] # Identifier is not written to output files
                            accumulated_mapudungun = ""
                            accumulated_spanish = ""
                        elif cleaned_line.startswith("M:"):
                            fragment = cleaned_line[2:].strip()
                            cleaned_fragment = remove_tags(fragment) # As per your code, this includes tags
                            accumulated_mapudungun += cleaned_fragment + " "
                        elif cleaned_line.startswith("C:"):
                            fragment = cleaned_line[2:].strip()
                            cleaned_fragment = remove_tags(fragment) # As per your code, this includes tags
                            accumulated_spanish += cleaned_fragment + " "

                    print(f"--- Finished processing file: {relative_path} ({file_lines_processed} lines) ---")

                except UnicodeEncodeError as e:
                    print(f"\nERROR: Encoding issue writing output for identifier '{current_identifier}' from file '{relative_path}' near line {file_lines_processed}.", file=sys.stderr)
                    print(f"Output encoding '{output_encoding}' failed. Details: {e}", file=sys.stderr)
                    raise
                except Exception as e:
                    print(f"\nERROR: Unexpected error processing decoded content from '{relative_path}' near line {file_lines_processed}. Skipping rest of file.", file=sys.stderr)
                    print(f"Error details: {e}", file=sys.stderr)
                    import traceback
                    traceback.print_exc(file=sys.stderr)
                    current_identifier = None
                    accumulated_mapudungun = ""
                    accumulated_spanish = ""
                    continue

            # --- After processing ALL files ---
            if current_identifier is not None and (accumulated_mapudungun or accumulated_spanish):
                 try:
                    # Write the last accumulated pair
                    map_outfile.write(accumulated_mapudungun.strip() + "\n")
                    es_outfile.write(accumulated_spanish.strip() + "\n")
                    lines_written += 1
                 except UnicodeEncodeError as e:
                     print(f"\nERROR: Encoding issue writing final output for identifier '{current_identifier}'. {e}", file=sys.stderr)
                     raise

        print(f"\nProcessing complete!")
        print(f"Processed {files_processed_count} '.txt' files ({total_lines_processed} lines total).")
        print(f"Wrote {lines_written} parallel lines to output files based on '{output_basename}'.")

    except IOError as e:
        print(f"\nERROR: Could not open output files (e.g., '{mapudungun_output_filepath}') or access input directory '{input_directory}'. {e}", file=sys.stderr)
    except LookupError as e:
        print(f"\nERROR: Invalid encoding name specified (likely output encoding). {e}", file=sys.stderr)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)

# --- How to use the script ---
# Usage: python extract_to_parallel.py --input <dir> --output <basename>
# Example:
#   python extract_to_parallel.py \
#     --input /home/it238/nobackup/autodelete/mapudungun/data \
#     --output /home/it238/nobackup/autodelete/mapudungun/data-processed/parallel_corpus

import argparse

parser = argparse.ArgumentParser(
    description='Extract parallel Mapudungun–Spanish text from M:/C: transcript files.'
)
parser.add_argument('--input', required=True,
                    help='Root directory containing .txt transcript files (searched recursively)')
parser.add_argument('--output', required=True,
                    help='Base path for output files (e.g. /path/to/parallel_corpus '
                         'produces parallel_corpus.arn and parallel_corpus.es)')
parser.add_argument('--encoding', default='utf-8',
                    help='Output encoding (default: utf-8)')
args = parser.parse_args()

process_directory_to_parallel_texts(args.input, args.output, output_encoding=args.encoding)