from bs4 import BeautifulSoup
import csv
import os

def read_html(file_path, output_dir , output_file_name):
    output_file_path = os.path.join(output_dir, output_file_name)
    with open(file_path, 'r', encoding='utf-8') as f:
        html_content = f.read()

    soup = BeautifulSoup(html_content, "html.parser")
    table = soup.find("table")
    rows = table.find_all("tr")

    with open(output_file_path , "w", newline="", encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile)
        for row in rows:
            data = row.find_all(["th", "td"])
            row_data = [cell.get_text(strip=True) for cell in data]
            csvwriter.writerow(row_data)

    print(f"CSV file '{output_file_name}' created successfully at '{output_dir}'")

# Run for SCORING file
read_html(file_path = "/Users/s190387/Desktop/golf/data/lpga_kpmg.txt", 
          output_dir= "/Users/s190387/Desktop/golf/data",
          output_file_name = "lpga_kpmg.csv")

# Run for STROKES GAINED file
read_html(file_path = "/Users/s190387/Desktop/golf/data/lpga_shotsgained.txt", 
          output_dir= "/Users/s190387/Desktop/golf/data",
          output_file_name = "lpga_shotsgained.csv")

# Run for DRIVING file
read_html(file_path = "/Users/s190387/Desktop/golf/data/lpga_driving.txt", 
          output_dir= "/Users/s190387/Desktop/golf/data",
          output_file_name = "lpga_driving.csv")

# Run for APPROACH file
read_html(file_path = "/Users/s190387/Desktop/golf/data/lpga_approach.txt", 
          output_dir= "/Users/s190387/Desktop/golf/data",
          output_file_name = "lpga_approach.csv")
# Run for SHORTGAME/PUTTING file
read_html(file_path = "/Users/s190387/Desktop/golf/data/lpga_shortgame.txt", 
          output_dir= "/Users/s190387/Desktop/golf/data",
          output_file_name = "lpga_shortgame.csv")



