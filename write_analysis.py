html = open("templates/analysis.html", encoding="utf-8").read()

start_marker = "<!-- Confusion matrix -->"
end_marker   = "<!-- Per-class metrics -->"

start_idx = html.find(start_marker)
end_idx   = html.find(end_marker)

new_section = """<!-- Overall 7x7 Confusion Matrix -->
<section class="glass-card chart-card fade-in">
    <h2 class="chart-title"><i class="fa-solid fa-table-cells"></i> Overall Confusion Matrix &mdash; All 7 Classes</h2>
    <p class="chart-sub">
        Rows = True class &nbsp;|&nbsp; Columns = Predicted class &nbsp;|&nbsp;
        <span style="color:var(--green)">&#9632;</span> Diagonal = correct &nbsp;|&nbsp;
        <span style="color:var(--red)">&#9632;</span> Off-diagonal = misclassification.
        Each cell shows count and row-recall %.
    </p>
    <div class="cm-table-wrap">
        <table class="cm-table">
            <thead>
                <tr>
                    <th class="cm-corner">True &#8595; / Pred &#8594;</th>
                    <th class="cm-head">AMD</th><th class="cm-head">DME</th><th class="cm-head">ERM</th>
                    <th class="cm-head">NO</th><th class="cm-head">RAO</th><th class="cm-head">RVO</th>
                    <th class="cm-head">VID</th>
                    <th class="cm-head cm-total-head">Total</th>
                    <th class="cm-head cm-recall-head">Recall</th>
                </tr>
            </thead>
            <tbody>
                {% set cm_real  = [[167,1,5,4,1,4,2],[1,14,2,1,0,2,2],[5,0,13,4,0,0,1],[3,1,0,45,0,0,0],[0,0,0,0,3,0,0],[1,1,1,3,0,9,0],[0,0,2,1,0,0,8]] %}
                {% set cls_list = ["AMD","DME","ERM","NO","RAO","RVO","VID"] %}
                {% set row_tots = [184,22,23,49,3,15,11] %}
                {% set recalls  = [90.8,63.6,56.5,91.8,100.0,60.0,72.7] %}
                {% set clr      = ["amber","pink","cyan","green","red","violet","orange"] %}
                {% for i in range(7) %}
                <tr>
                    <td class="cm-row-label"><span class="badge badge--{{ clr[i] }}">{{ cls_list[i] }}</span></td>
                    {% for j in range(7) %}
                    {% set v = cm_real[i][j] %}
                    {% set p = (v / row_tots[i] * 100) | round(1) %}
                    <td class="cm-cell {% if i==j %}cm-diag{% elif v>0 %}cm-err{% endif %}" title="{{ cls_list[i] }} predicted as {{ cls_list[j] }}: {{ v }} ({{ p }}%)">
                        <span class="cm-count">{{ v }}</span><span class="cm-pct">{{ p }}%</span>
                    </td>
                    {% endfor %}
                    <td class="cm-total">{{ row_tots[i] }}</td>
                    <td class="cm-recall {% if recalls[i]>=80 %}cm-recall--good{% elif recalls[i]>=60 %}cm-recall--fair{% else %}cm-recall--poor{% endif %}">{{ recalls[i] }}%</td>
                </tr>
                {% endfor %}
            </tbody>
            <tfoot>
                <tr>
                    <td class="cm-corner" style="font-size:.78rem;color:var(--muted);">Precision &#8594;</td>
                    {% for p in [94,82,57,78,75,60,62] %}
                    <td class="cm-precision {% if p>=80 %}cm-recall--good{% elif p>=60 %}cm-recall--fair{% else %}cm-recall--poor{% endif %}">{{ p }}%</td>
                    {% endfor %}
                    <td class="cm-total" style="font-weight:700;">307</td>
                    <td class="cm-recall cm-recall--good" style="font-weight:700;">84.4%</td>
                </tr>
            </tfoot>
        </table>
    </div>
    <div class="cm-legend">
        <span class="cm-legend-item"><span class="cm-legend-box cm-legend-diag"></span> Correct prediction</span>
        <span class="cm-legend-item"><span class="cm-legend-box cm-legend-err"></span> Misclassification</span>
        <span class="cm-legend-item"><span class="cm-legend-box cm-legend-zero"></span> Zero</span>
    </div>
    <div style="margin-top:1.5rem;">
        <p style="color:var(--muted);font-size:.82rem;margin-bottom:.6rem;"><i class="fa-solid fa-image"></i> High-resolution heatmap (matplotlib):</p>
        <img src="{{ url_for('static', filename='plots/confusion_matrix.png') }}" alt="Confusion matrix" class="training-img" onerror="this.parentElement.style.display='none'">
    </div>
</section>

<!-- Individual Binary Confusion Matrices -->
<section class="glass-card fade-in">
    <h2 class="chart-title"><i class="fa-solid fa-grid-2"></i> Individual Binary Confusion Matrices &mdash; One vs Rest</h2>
    <p class="chart-sub">
        Each card shows a 2&times;2 binary confusion matrix for one disease vs all others.
        <strong style="color:var(--green)">TP</strong> = True Positive &nbsp;
        <strong style="color:var(--red)">FN</strong> = False Negative &nbsp;
        <strong style="color:var(--amber)">FP</strong> = False Positive &nbsp;
        <strong style="color:var(--muted)">TN</strong> = True Negative
    </p>
    {% set diseases = [
        ("AMD","amber","Age-Related Macular Degeneration",167,17,10,113,90.8,94.4,92.5),
        ("DME","pink","Diabetic Macular Edema",14,8,3,282,63.6,82.4,71.8),
        ("ERM","cyan","Epiretinal Membrane",13,10,10,274,56.5,56.5,56.5),
        ("NO","green","Normal Retina",45,4,5,253,91.8,90.0,90.9),
        ("RAO","red","Retinal Artery Occlusion",3,0,1,303,100.0,75.0,85.7),
        ("RVO","violet","Retinal Vein Occlusion",9,6,6,286,60.0,60.0,60.0),
        ("VID","orange","Vitreomacular Interface Disease",8,3,5,291,72.7,61.5,66.7)
    ] %}
    <div class="binary-cm-grid">
        {% for code,color,fullname,tp,fn,fp,tn,recall,prec,f1 in diseases %}
        <div class="binary-cm-card">
            <div class="binary-cm-header binary-cm-header--{{ color }}">
                <span class="badge badge--{{ color }}">{{ code }}</span>
                <div>
                    <div style="font-weight:700;font-size:.92rem;">{{ fullname }}</div>
                    <div style="font-size:.75rem;color:var(--muted);margin-top:.1rem;">{{ code }} vs All Others (n={{ tp+fn+fp+tn }})</div>
                </div>
            </div>
            <div class="binary-cm-body">
                <div class="bcm-axis-x">Predicted Label</div>
                <div class="bcm-axis-y">True Label</div>
                <div class="bcm-grid">
                    <div class="bcm-blank"></div>
                    <div class="bcm-col-head bcm-col-pos">{{ code }}</div>
                    <div class="bcm-col-head bcm-col-neg">Other</div>
                    <div class="bcm-row-head bcm-row-pos">{{ code }}</div>
                    <div class="bcm-cell bcm-tp" title="TP: {{ code }} correctly detected">
                        <span class="bcm-tag">TP</span>
                        <span class="bcm-num">{{ tp }}</span>
                        <span class="bcm-pct">{{ ((tp/(tp+fn))*100)|round(1) }}%</span>
                    </div>
                    <div class="bcm-cell bcm-fn" title="FN: {{ code }} missed, predicted as Other">
                        <span class="bcm-tag">FN</span>
                        <span class="bcm-num">{{ fn }}</span>
                        <span class="bcm-pct">{{ ((fn/(tp+fn))*100)|round(1) }}%</span>
                    </div>
                    <div class="bcm-row-head bcm-row-neg">Other</div>
                    <div class="bcm-cell bcm-fp" title="FP: Other incorrectly predicted as {{ code }}">
                        <span class="bcm-tag">FP</span>
                        <span class="bcm-num">{{ fp }}</span>
                        <span class="bcm-pct">{{ ((fp/(fp+tn))*100)|round(1) }}%</span>
                    </div>
                    <div class="bcm-cell bcm-tn" title="TN: Other correctly rejected">
                        <span class="bcm-tag">TN</span>
                        <span class="bcm-num">{{ tn }}</span>
                        <span class="bcm-pct">{{ ((tn/(fp+tn))*100)|round(1) }}%</span>
                    </div>
                </div>
            </div>
            <div class="bcm-metrics">
                <div class="bcm-metric">
                    <span class="bcm-metric-lbl">Precision</span>
                    <span class="bcm-metric-val {% if prec>=80 %}cm-recall--good{% elif prec>=60 %}cm-recall--fair{% else %}cm-recall--poor{% endif %}">{{ prec }}%</span>
                </div>
                <div class="bcm-metric">
                    <span class="bcm-metric-lbl">Recall</span>
                    <span class="bcm-metric-val {% if recall>=80 %}cm-recall--good{% elif recall>=60 %}cm-recall--fair{% else %}cm-recall--poor{% endif %}">{{ recall }}%</span>
                </div>
                <div class="bcm-metric">
                    <span class="bcm-metric-lbl">F1-Score</span>
                    <span class="bcm-metric-val {% if f1>=80 %}cm-recall--good{% elif f1>=60 %}cm-recall--fair{% else %}cm-recall--poor{% endif %}">{{ f1 }}%</span>
                </div>
                <div class="bcm-metric">
                    <span class="bcm-metric-lbl">Support</span>
                    <span class="bcm-metric-val" style="color:var(--muted);">{{ tp+fn }}</span>
                </div>
            </div>
        </div>
        {% endfor %}
    </div>
    <div class="binary-cm-legend">
        <span class="binary-cm-legend-item"><span class="bcm-box bcm-tp"></span> TP &ndash; Correctly detected</span>
        <span class="binary-cm-legend-item"><span class="bcm-box bcm-fn"></span> FN &ndash; Missed</span>
        <span class="binary-cm-legend-item"><span class="bcm-box bcm-fp"></span> FP &ndash; False alarm</span>
        <span class="binary-cm-legend-item"><span class="bcm-box bcm-tn"></span> TN &ndash; Correctly rejected</span>
    </div>
</section>

"""

new_html = html[:start_idx] + new_section + html[end_idx:]
open("templates/analysis.html", "w", encoding="utf-8").write(new_html)
print("DONE - wrote", len(new_html), "chars")
