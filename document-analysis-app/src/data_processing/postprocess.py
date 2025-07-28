# def format_results(results):
#     # Format the results for better readability
#     formatted_results = []
#     for result in results:
#         formatted_results.append(f"Document ID: {result['id']}, Score: {result['score']}")
#     return formatted_results

# def save_results(formatted_results, output_file):
#     # Save the formatted results to a specified output file
#     with open(output_file, 'w') as f:
#         for line in formatted_results:
#             f.write(line + '\n')

# def postprocess_data(analyzed_data, output_file):
#     # Postprocess the analyzed data and save the results
#     formatted_results = format_results(analyzed_data)
#     save_results(formatted_results, output_file)