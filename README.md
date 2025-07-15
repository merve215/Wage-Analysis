# Employment-Weighted Wage Matrix Generator with Strategic Hierarchy

This project generates robust, employment-weighted wage matrices for multiple countries and years using official wage and employment data. It employs a strategic hierarchy to impute missing values, ensuring the most reliable wage estimates possible for each economic activity × occupation cell.

## Data Filtering and Preprocessing

- **Currency:** Only data denominated in U.S. dollars is considered to ensure comparability across countries.
- **Sex:** Only “Total” sex (all sexes combined) is used to avoid double counting and to provide aggregate figures.
- **Regional Groupings:** For missing data, the script uses regional imputation, sourcing values from predefined groups of similar countries (see `SIMILAR_COUNTRIES` in the code).

## Features

- **Employment-weighted wage imputation** using direct data when available.
- **Strategic hierarchy** for filling missing values, including geometric means, regional imputation, and global medians.
- **Quality scoring** per country-year, summarizing data robustness.
- **Cell-level strategy logging** for transparency into how each wage value was derived.
- **Long-format CSV output** with full provenance for downstream analysis.

## Input Data

Place the following CSV files in the root directory:

- `Average monthly earnings of employees by sex and economic activity.csv`
- `Average monthly earnings of employees by sex and occupation.csv`
- `Employment by economic activity and occupation (thousands).csv`

## How It Works

1. **Data Loading and Cleaning:**  
   The script loads activity, occupation, and employment data, and filters for "Total" sex and USD currency.

2. **Matrix Construction:**  
   For each country-year:
   - Builds a wage matrix (economic activities × occupations).
   - Fills each cell using the most robust available method:
     - **Employment-weighted average** (best).
     - **Geometric mean** (if both wages exist).
     - **Activity or occupation wage only** (if only one exists).
     - **Regional imputation** (from similar countries).
     - **Global median** (last resort).

3. **Quality Analysis:**  
   - Logs the imputation strategy for every cell.
   - Computes a `Quality_Score` for each country-year, summarizing overall data robustness.

4. **Export:**  
   - Outputs a long-format CSV with columns:
     - `Country, Year, Economic_activity, Occupation, Wage, Strategy, Quality_Score`
   - `Strategy` indicates the method used for each cell.
   - `Quality_Score` is the same for all rows of a country-year and summarizes the matrix's overall quality.

## Example Output

| Country | Year | Economic_activity | Occupation                | Wage      | Strategy             | Quality_Score |
|---------|------|------------------|---------------------------|-----------|----------------------|---------------|
| Uganda  | 2017 | Agriculture      | Skill levels 3 and 4 ~ high | 105.83 | employment_weighted  | 86.7          |
| Uganda  | 2017 | Industry         | Skill levels 3 and 4 ~ high | 153.52 | geometric_mean       | 86.7          |

## Usage

1. Ensure all required CSV files are present.
2. Run the main processing script.
3. The output will be saved as `all_wage_matrices_with_quality_and_strategy.csv`.

## Customization

- Regional groupings for imputation can be expanded in the `SIMILAR_COUNTRIES` dictionary in the script.
- Weights for quality scoring can be adjusted in the `quality_weights` dictionary.

## License

MIT License

## Author

- [merve215](https://github.com/merve215)