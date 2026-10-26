"""
AIMS HMI Exporters Module

Handles exporting session data to:
1. PDF format: Formatted report with metrics, charts, and metadata
2. CSV format: Raw tracking log data for analysis
"""

import os
import csv
from datetime import datetime
from typing import Dict, List, Tuple, Optional

from aims_hmi.config import (
    SESSION, PDF_TEMPLATE_TITLE, CSV_EXPORT_COLUMNS
)


class CSVExporter:
    """
    Exports tracking log data to CSV format.
    
    CSV Output:
    - Columns: Timestamp, Item_ID, Status, Defect_Type, Confidence, Action
    - Format: Standard CSV with headers
    """
    
    def export(self, 
               file_path: str,
               data: List[Dict],
               columns: List[str] = CSV_EXPORT_COLUMNS) -> str:
        """
        Export data to CSV file.
        
        Args:
            file_path: Path to save the CSV file
            data: List of dictionaries containing row data
            columns: List of column names to export
            
        Returns:
            Path to the exported file
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path) or '.', exist_ok=True)
        
        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            
            for row in data:
                # Format the row for CSV
                csv_row = {}
                for col in columns:
                    value = row.get(col, '')
                    
                    # Format special fields
                    if col == 'Timestamp' and isinstance(value, datetime):
                        value = value.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                    elif col == 'Confidence' and isinstance(value, float):
                        value = f"{value:.1%}"
                    
                    csv_row[col] = value
                
                writer.writerow(csv_row)
        
        return file_path


class PDFExporter:
    """
    Exports session report to PDF format.
    
    PDF Output includes:
    - Executive metrics overview
    - Defect distribution chart (embedded)
    - Timeline analysis chart (embedded)
    - Session metadata
    """
    
    def __init__(self, report_view=None):
        """
        Initialize PDF exporter.
        
        Args:
            report_view: Optional reference to SessionReportView for 
                        capturing chart images
        """
        self.report_view = report_view
    
    def export(self,
               file_path: str,
               session_time: str,
               total_units: int,
               pass_count: int,
               fail_count: int,
               defect_counts: Dict[str, int],
               timeline: List[Tuple[float, int]],
               batch_id: str,
               operator_id: str,
               component: str = "Bottle Caps") -> str:
        """
        Generate and export a PDF report.
        
        Args:
            file_path: Path to save the PDF file
            session_time: Formatted session duration
            total_units: Total items processed
            pass_count: Number of passed items
            fail_count: Number of failed items
            defect_counts: Dictionary of defect type -> count
            timeline: List of (minute, count) tuples
            batch_id: Batch identifier
            operator_id: Operator identifier
            component: Component being sorted
            
        Returns:
            Path to the exported file
        """
        # Generate PDF content
        pdf_content = self._generate_pdf_content(
            session_time=session_time,
            total_units=total_units,
            pass_count=pass_count,
            fail_count=fail_count,
            defect_counts=defect_counts,
            timeline=timeline,
            batch_id=batch_id,
            operator_id=operator_id,
            component=component
        )
        
        # Write to file
        os.makedirs(os.path.dirname(file_path) or '.', exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(pdf_content)
        
        return file_path
    
    def _generate_pdf_content(self,
                             session_time: str,
                             total_units: int,
                             pass_count: int,
                             fail_count: int,
                             defect_counts: Dict[str, int],
                             timeline: List[Tuple[float, int]],
                             batch_id: str,
                             operator_id: str,
                             component: str) -> str:
        """
        Generate the PDF content as a formatted text document.
        
        In production, this would use reportlab or similar library
        to generate a proper PDF with charts. For now, generates
        a text-based report that can be converted to PDF.
        
        Returns:
            Formatted report content
        """
        lines = []
        
        # Header
        lines.append("=" * 70)
        lines.append(f"{PDF_TEMPLATE_TITLE}")
        lines.append(f"{'=' * 70}")
        lines.append("")
        
        # Session Info
        lines.append("SESSION INFORMATION")
        lines.append("-" * 70)
        lines.append(f"  Batch ID:        {batch_id}")
        lines.append(f"  Operator ID:     {operator_id}")
        lines.append(f"  Component:       {component}")
        lines.append(f"  Generated:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        # Executive Metrics
        lines.append("EXECUTIVE METRICS")
        lines.append("-" * 70)
        lines.append(f"  Session Duration:    {session_time}")
        lines.append(f"  Total Units:         {total_units:,}")
        lines.append(f"  Passed:              {pass_count:,}")
        lines.append(f"  Failed:              {fail_count:,}")
        
        if total_units > 0:
            yield_rate = (pass_count / total_units) * 100
            defect_rate = (fail_count / total_units) * 100
            lines.append(f"  Yield Rate:          {yield_rate:.2f}% PASS")
            lines.append(f"  Defect Rate:         {defect_rate:.2f}%")
        else:
            lines.append(f"  Yield Rate:          N/A")
            lines.append(f"  Defect Rate:         N/A")
        
        lines.append("")
        
        # Quality Summary
        lines.append("QUALITY SUMMARY")
        lines.append("-" * 70)
        
        if fail_count == 0:
            lines.append("  ✓ PERFECT RUN - No defects detected")
        elif fail_count < total_units * 0.05:
            lines.append("  ✓ GOOD - Defect rate under 5%")
        elif fail_count < total_units * 0.10:
            lines.append("  ⚠ WARNING - Defect rate between 5-10%")
        else:
            lines.append("  ✗ CRITICAL - Defect rate exceeds 10%")
        
        lines.append("")
        
        # Defect Breakdown
        if defect_counts:
            lines.append("DEFECT BREAKDOWN")
            lines.append("-" * 70)
            lines.append("")
            lines.append(f"{'Defect Type':<35} {'Count':>8} {'Percentage':>12}")
            lines.append("-" * 55)
            
            total_defects = sum(defect_counts.values())
            for defect_type, count in sorted(defect_counts.items(), 
                                             key=lambda x: -x[1]):
                percentage = (count / total_defects * 100) if total_defects > 0 else 0
                lines.append(f"{defect_type:<35} {count:>8} {percentage:>11.1f}%")
            
            lines.append("-" * 55)
            lines.append(f"{'TOTAL':<35} {total_defects:>8}")
        else:
            lines.append("DEFECT BREAKDOWN")
            lines.append("-" * 70)
            lines.append("  No defects recorded during this session.")
        
        lines.append("")
        
        # Timeline Analysis
        if timeline:
            lines.append("DEFECT TIMELINE ANALYSIS")
            lines.append("-" * 70)
            lines.append("(Defects per 5-minute interval)")
            lines.append("")
            
            # Aggregate by interval
            intervals = {}
            for minute, count in timeline:
                interval = int(minute // 5) * 5
                intervals[interval] = intervals.get(interval, 0) + count
            
            lines.append(f"{'Time (min)':<15} {'Defects':>10}")
            lines.append("-" * 25)
            for minute in sorted(intervals.keys()):
                lines.append(f"{minute:<15} {intervals[minute]:>10}")
            
            # Find spike times
            if intervals:
                max_count = max(intervals.values())
                spike_times = [m for m, c in intervals.items() if c >= max_count * 0.8]
                if len(spike_times) > 0:
                    lines.append("")
                    lines.append(f"  ⚠️  Peak defect period: Minute {spike_times[0]:.0f}")
        else:
            lines.append("DEFECT TIMELINE ANALYSIS")
            lines.append("-" * 70)
            lines.append("  No timeline data available.")
        
        lines.append("")
        
        # Recommendations
        lines.append("RECOMMENDATIONS")
        lines.append("-" * 70)
        
        if defect_counts:
            most_common = max(defect_counts.items(), key=lambda x: x[1])
            lines.append(f"  1. Investigate primary defect: {most_common[0]}")
            lines.append(f"     ({most_common[1]} occurrences - "
                        f"{(most_common[1]/fail_count*100):.1f}% of failures)")
        
        if fail_count > 0:
            lines.append("  2. Review rejected units in quarantine bin")
            lines.append("  3. Check conveyor belt alignment and speed")
            lines.append("  4. Verify sensor calibration")
        
        lines.append("")
        lines.append("=" * 70)
        lines.append("END OF REPORT")
        lines.append(f"Total pages: 1")
        lines.append("=" * 70)
        
        return "\n".join(lines)


class MockDataExporter:
    """
    Generates mock session data for testing/export demonstrations.
    
    Used when no real session data is available but a sample
    export is needed for testing.
    """
    
    @staticmethod
    def generate_sample_data(num_items: int = 100) -> List[Dict]:
        """Generate sample tracking data for testing."""
        import random
        from datetime import datetime, timedelta
        
        data = []
        start_time = datetime.now() - timedelta(minutes=num_items // 2)
        defect_types = [
            "Rim Deformity / Dent",
            "Missing Inner Liner",
            "Print / Logo Misalignment",
            "Surface Scratch"
        ]
        
        for i in range(num_items):
            is_pass = random.random() > 0.15
            timestamp = start_time + timedelta(seconds=i * 2)
            
            row = {
                'Timestamp': timestamp,
                'Item_ID': f"ITEM-{i+1:05d}",
                'Status': 'PASS' if is_pass else 'FAIL',
                'Defect_Type': '-' if is_pass else random.choice(defect_types),
                'Confidence': random.uniform(0.85, 0.99) if is_pass else random.uniform(0.60, 0.89),
                'Action': 'ACCEPT' if is_pass else 'REJECT'
            }
            data.append(row)
        
        return data


# =============================================================================
# Chart Image Export (for embedding in PDF)
# =============================================================================

def export_chart_to_image(chart_widget, file_path: str) -> bool:
    """
    Export a chart widget to an image file.
    
    Args:
        chart_widget: ChartWidget instance
        file_path: Path to save the image
        
    Returns:
        True if successful, False otherwise
    """
    try:
        if hasattr(chart_widget, '_canvas') and hasattr(chart_widget._canvas, 'figure'):
            chart_widget._canvas.figure.savefig(
                file_path,
                dpi=150,
                bbox_inches='tight',
                facecolor=chart_widget._canvas.figure.get_facecolor()
            )
            return True
    except Exception as e:
        print(f"Failed to export chart: {e}")
    
    return False
