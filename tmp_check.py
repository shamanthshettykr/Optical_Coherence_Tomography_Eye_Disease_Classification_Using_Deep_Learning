# Write new analysis.html
html = open('templates/analysis.html', encoding='utf-8').read()

# Find the section to replace - from confusion matrix comment to per-class metrics
start_marker = '<!-- Confusion matrix -->'
end_marker = '<!-- Per-class metrics -->'

start_idx = html.find(start_marker)
end_idx = html.find(end_marker)

print(f'start={start_idx}, end={end_idx}')
if start_idx == -1 or end_idx == -1:
    print('MARKERS NOT FOUND')
else:
    print('Found both markers OK')
