import openpyxl

# Load the workbook and select the active worksheet
workbook = openpyxl.load_workbook('dataset_changed_gpt-4-1106-preview_hn-2_mdt-2_7.xlsx')
worksheet = workbook.active

# Iterate over all cells in all rows
for row in worksheet.iter_rows(values_only=False):
    for cell in row:
        # If the cell contains 'nan' (case insensitive), replace it with an empty string
        if str(cell.value).lower() == 'nan':
            cell.value = ''

# Save the modified workbook to 'dataset_82.xlsx'
workbook.save('dataset_82.xlsx')