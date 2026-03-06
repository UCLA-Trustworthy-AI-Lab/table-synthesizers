import os
import warnings
import shutil

  
class Saver:
  def __init__(self, reportPath='test_results') -> None:
    self.reportPath = reportPath
    
  def saveReport(self, report):
    """
        Saves the report, including tables and plots.

        Args:
        report (dict): Dictionary of evaluation metrics and plots.
    """
    # Ensure the report save folder exist
    if not os.path.exists(self.reportPath):
      print("Making report path at:",self.reportPath)
      os.mkdir(self.reportPath)
    # Clear any prior contents
    elif os.path.exists(self.reportPath) and os.path.isdir(self.reportPath):
        try:
          shutil.rmtree(self.reportPath)
          os.mkdir(self.reportPath)
        except:
          pass

    # Save all overall metrics in one table
    
    # Save columns-wise metrics in seperate tables
    if 'tables' in report:
        self.saveTables(report['tables'])
    else:
        warnings.warn("No table to save!")
    # Save plots: 
    if 'plots' in report:
        self.savePlots(report['plots'])
        pass
    else:
        warnings.warn("No plot to save!")
    
  def saveTables(self, tables):
    """
        Saves the tables from the report.

        Args:
        tables (dict): Dictionary of tables to be saved.
    """
    for tableType in tables:
      try:
        if isinstance(tables[tableType],dict):
          for var in tables[tableType]:
            complete_table_name = "_".join([tableType, var]) + '.csv'
            try:
              tables[tableType][var].to_csv(os.path.join(self.reportPath, complete_table_name),index=False)
            except Exception as e:
              print("Error converting ", complete_table_name, " to csv. Error: ", e)
              #print("tables[tableType][var] is:",tables[tableType][var], tableType,var)
              #print("tables[tableType]",tables[tableType])
        else:
          complete_table_name = ".".join([tableType,'csv'])
          try:
            tables[tableType].to_csv(os.path.join(self.reportPath, complete_table_name),index=False)
          except:
            print("Error converting ", complete_table_name, " to csv. Error: ", e)

      except Exception as e:
        warnings.warn(f"Error {e} occured when saving {tableType}!")


  def savePlots(self, plots):
    """
    Saves the plots from the report.

    Args:
        plots (dict): Dictionary of plots to be saved.
    """
    for plotType in plots:

        # Some entries may be missing / unplotable
        if plotType is None or plots[plotType] is None:
            print(f"No available plotter for plot type: {plotType}. Skipping.")
            continue

        # If there is one subplot for each variable:
        if isinstance(plots[plotType], dict):
            for var in plots[plotType]:
                fig = plots[plotType][var]
                if fig is None:
                    continue

                png_name = "_".join([str(plotType), str(var)]) + ".png"
                png_path = os.path.join(self.reportPath, png_name)

                try:
                    fig.write_image(png_path)
                except Exception as e:
                    # Fallback to HTML (doesn't require kaleido)
                    html_name = "_".join([str(plotType), str(var)]) + ".html"
                    html_path = os.path.join(self.reportPath, html_name)
                    try:
                        fig.write_html(html_path)
                        print(f"[WARN] PNG export failed ({type(e).__name__}: {e}). Saved HTML: {html_name}")
                    except Exception as e2:
                        print(f"[WARN] PNG+HTML export failed for {plotType}_{var} "
                              f"({type(e2).__name__}: {e2}). Skipping.")

        # If there is just one plot for the whole table:
        else:
            fig = plots[plotType]

            png_name = ".".join([str(plotType), "png"])
            png_path = os.path.join(self.reportPath, png_name)

            try:
                fig.write_image(png_path)
            except Exception as e:
                html_name = ".".join([str(plotType), "html"])
                html_path = os.path.join(self.reportPath, html_name)
                try:
                    fig.write_html(html_path)
                    print(f"[WARN] PNG export failed ({type(e).__name__}: {e}). Saved HTML: {html_name}")
                except Exception as e2:
                    print(f"[WARN] PNG+HTML export failed for {plotType} "
                          f"({type(e2).__name__}: {e2}). Skipping.")


