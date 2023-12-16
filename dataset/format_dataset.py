import openpyxl

workbook = openpyxl.load_workbook('dataset_changed_gpt-4-1106-preview_hn-2_mdt-2_7.xlsx')
worksheet = workbook.active

for row in worksheet.iter_rows(values_only=False):
    for cell in row:
        if str(cell.value).lower() == 'nan':
            cell.value = ''

workbook.save('dataset_82.xlsx')