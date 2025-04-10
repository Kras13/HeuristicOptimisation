import csv
import re

input_file = "original_data.csv"

output_file = "cleaned_data.csv"

date_pattern = re.compile(r"\d{4}-\d{2}-\d{2}")

with open(input_file, "r", newline='', encoding="utf-8") as infile:
    reader = csv.reader(infile)
    rows = list(reader)

start_index = None
for i, row in enumerate(rows):
    if row and date_pattern.match(row[0]):
        start_index = i
        break

if start_index is not None:
    with open(output_file, "w", newline='', encoding="utf-8") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(["Date", "Close", "High", "Low", "Open", "Volume"])
        for row in rows[start_index:]:
            if row and len(row) == 6:
                writer.writerow(row)
    print(f"Данните са почистени и записани в {output_file}")
else:
    print("Не е намерен валиден ред с дата.")
