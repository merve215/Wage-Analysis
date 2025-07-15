import pandas as pd
import numpy as np
import os


def process_batch_with_strategic_hierarchy():
    """
    Optimized batch processing with Strategic Hierarchy missing value handling
    Current Date and Time (UTC - YYYY-MM-DD HH:MM:SS formatted): 2025-06-30 10:13:30
    Current User's Login: merve215
    """
    print("=== Employment-Weighted Wage Matrix Generator with Strategic Hierarchy ===")
    print(f"Current Date and Time (UTC - YYYY-MM-DD HH:MM:SS formatted): 2025-06-30 10:13:30")
    print(f"Current User's Login: merve215")
    print("=" * 80)

    # 🔧 Updated file paths for your data structure
    activity_wage_file = "Average monthly earnings of employees by sex and economic activity.csv"
    occupation_wage_file = "Average monthly earnings of employees by sex and occupation.csv"
    employment_file = "Employment by economic activity and occupation (thousands).csv"

    # Memory-optimized loading
    print("Loading data with memory optimization...")
    df_activity = pd.read_csv(activity_wage_file)
    df_occupation = pd.read_csv(occupation_wage_file)
    df_employment = pd.read_csv(employment_file)

    print(
        f"Raw data loaded: Activity {len(df_activity):,}, Occupation {len(df_occupation):,}, Employment {len(df_employment):,}")

    # 🔧 Process your actual column structure
    # Filter for Total sex only (to avoid double counting)
    df_activity = df_activity[df_activity['sex.label'] == 'Total'].copy()
    df_occupation = df_occupation[df_occupation['sex.label'] == 'Total'].copy()

    # Extract clean activity and occupation names
    df_activity['Economic_activity'] = df_activity['classif1.label'].str.replace(r'Economic activity \([^)]+\): ', '',
                                                                                 regex=True)
    df_occupation['Occupation'] = df_occupation['classif1.label'].str.replace(r'Occupation \([^)]+\): ', '', regex=True)
    df_employment['Economic_activity'] = df_employment['classif1.label'].str.replace(r'Economic activity \([^)]+\): ',
                                                                                     '', regex=True)
    df_employment['Occupation'] = df_employment['classif2.label'].str.replace(r'Occupation \([^)]+\): ', '', regex=True)

    # Filter for USD currency
    currency_choice = 'U.S. dollars'
    df_activity = df_activity[df_activity['classif2.label'].str.contains(currency_choice, case=False, na=False)].copy()
    df_occupation = df_occupation[
        df_occupation['classif2.label'].str.contains(currency_choice, case=False, na=False)].copy()

    # Create essential columns mapping to your structure
    df_activity['Country'] = df_activity['ref_area.label']
    df_activity['Year'] = df_activity['time']
    df_activity['Value'] = df_activity['obs_value']

    df_occupation['Country'] = df_occupation['ref_area.label']
    df_occupation['Year'] = df_occupation['time']
    df_occupation['Value'] = df_occupation['obs_value']

    df_employment['Country'] = df_employment['ref_area.label']
    df_employment['Year'] = df_employment['time']
    df_employment['Value'] = df_employment['obs_value']

    # Remove 'Total' categories to avoid aggregation issues
    df_activity = df_activity[~df_activity['Economic_activity'].isin(['Total'])].copy()
    df_occupation = df_occupation[~df_occupation['Occupation'].isin(['Total'])].copy()
    df_employment = df_employment[~df_employment['Economic_activity'].isin(['Total'])].copy()
    df_employment = df_employment[~df_employment['Occupation'].isin(['Total'])].copy()

    print(f"✓ Activity data (USD, filtered): {len(df_activity):,} rows")
    print(f"✓ Occupation data (USD, filtered): {len(df_occupation):,} rows")
    print(f"✓ Employment data (filtered): {len(df_employment):,} rows")

    # 🔧 Expanded regional groupings - add more countries as needed
    SIMILAR_COUNTRIES = {
        'Aruba': ['Netherlands', 'Curacao', 'Sint Maarten', 'Barbados'],
        'United States': ['Canada', 'Australia', 'United Kingdom'],
        'Germany': ['France', 'Netherlands', 'Austria', 'Belgium'],
        'India': ['Bangladesh', 'Pakistan', 'Sri Lanka'],
        'Brazil': ['Argentina', 'Chile', 'Colombia', 'Mexico'],
        'Japan': ['South Korea', 'Singapore', 'Australia'],
        'France': ['Germany', 'Italy', 'Spain', 'Netherlands'],
        'China': ['India', 'Vietnam', 'Thailand'],
        'Mexico': ['Brazil', 'Colombia', 'Argentina'],
        'South Africa': ['Nigeria', 'Kenya', 'Ghana'],
        'Netherlands': ['Germany', 'Belgium', 'Denmark', 'Austria'],
        'Australia': ['Canada', 'United States', 'United Kingdom'],
        'Canada': ['United States', 'Australia', 'United Kingdom'],
        'United Kingdom': ['United States', 'Canada', 'Australia']
    }

    # Batch processing by country
    countries = set(df_activity['Country'].unique()) & set(df_occupation['Country'].unique()) & set(
        df_employment['Country'].unique())
    all_matrices = {}
    quality_report = {}

    print(f"\n🌍 Discovered {len(countries)} countries:")
    for country in sorted(list(countries)[:10]):  # Show first 10
        print(f"   • {country}")
    if len(countries) > 10:
        print(f"   ... and {len(countries) - 10} more countries")

    print(f"\nBatch processing {len(countries)} countries with Strategic Hierarchy...")

    successful_countries = 0
    failed_countries = []

    for i, country in enumerate(countries, 1):
        try:
            print(f"\n[{i:3d}/{len(countries)}] Processing {country}...")

            # Get country-specific data (batch processing)
            country_activity = df_activity[df_activity['Country'] == country]
            country_occupation = df_occupation[df_occupation['Country'] == country]
            country_employment = df_employment[df_employment['Country'] == country]

            years = set(country_activity['Year'].unique()) & set(country_occupation['Year'].unique()) & set(
                country_employment['Year'].unique())

            matrices_created_for_country = 0

            for year in years:
                year_activity = country_activity[country_activity['Year'] == year]
                year_occupation = country_occupation[country_occupation['Year'] == year]
                year_employment = country_employment[country_employment['Year'] == year]

                if len(year_activity) == 0 or len(year_occupation) == 0 or len(year_employment) == 0:
                    continue

                # Create matrices with Strategic Hierarchy
                wage_matrix, employment_matrix, strategy_log = create_strategic_wage_matrix(
                    year_activity, year_occupation, year_employment, country, year, SIMILAR_COUNTRIES, df_activity,
                    df_occupation
                )

                if wage_matrix is not None and len(wage_matrix) > 0:
                    # Store results
                    key = f"{country}_{year}"
                    all_matrices[key] = {
                        'wage_matrix': wage_matrix,
                        'employment_matrix': employment_matrix,
                        'strategy_log': strategy_log,
                        'country': country,
                        'year': year
                    }

                    # Quality analysis
                    quality_score = analyze_data_quality(strategy_log)
                    quality_report[key] = quality_score

                    matrices_created_for_country += 1
                    print(
                        f"    ✓ {year}: {wage_matrix.shape[0]}×{wage_matrix.shape[1]} matrix | Quality: {quality_score['quality_score']:.1f}%")

            if matrices_created_for_country > 0:
                successful_countries += 1
            else:
                failed_countries.append(country)
                print(f"    ⚠️  {country}: No matrices created")

        except Exception as e:
            failed_countries.append(country)
            print(f"    ❌ {country}: Error - {str(e)[:50]}...")

    # Summary report
    print(f"\n{'=' * 70}")
    print("📊 STRATEGIC HIERARCHY PROCESSING COMPLETE")
    print(f"{'=' * 70}")
    print(f"✅ Successfully processed: {successful_countries}/{len(countries)} countries")
    print(f"✅ Total matrices created: {len(all_matrices)}")

    if quality_report:
        avg_quality = np.mean([q['quality_score'] for q in quality_report.values()])
        print(f"✅ Average data quality: {avg_quality:.1f}%")

        strategy_summary = {}
        for report in quality_report.values():
            for strategy, count in report['strategy_counts'].items():
                strategy_summary[strategy] = strategy_summary.get(strategy, 0) + count

        print(f"\n📈 Missing Value Strategy Usage:")
        for strategy, count in sorted(strategy_summary.items(), key=lambda x: x[1], reverse=True):
            print(f"   {strategy}: {count:,} cells")

    if failed_countries:
        print(f"\n⚠️  Countries with issues ({len(failed_countries)}):")
        for country in failed_countries[:5]:  # Show first 5
            print(f"   • {country}")

    return all_matrices, quality_report


def create_strategic_wage_matrix(year_activity, year_occupation, year_employment, country, year, similar_countries,
                                 all_activity_df, all_occupation_df):
    """
    Create wage matrix using Strategic Hierarchy for missing values
    """
    activities = year_activity['Economic_activity'].unique()
    occupations = year_occupation['Occupation'].unique()

    # Remove any remaining 'Total' or aggregate categories
    activities = [a for a in activities if 'Total' not in str(a)]
    occupations = [o for o in occupations if 'Total' not in str(o)]

    if len(activities) == 0 or len(occupations) == 0:
        return None, None, {}

    wage_matrix = pd.DataFrame(index=activities, columns=occupations, dtype=float)
    employment_matrix = pd.DataFrame(index=activities, columns=occupations, dtype=float)
    strategy_log = {}

    # Fill employment matrix
    for _, row in year_employment.iterrows():
        activity = row['Economic_activity']
        occupation = row['Occupation']
        employment_count = row['Value']

        if activity in activities and occupation in occupations:
            employment_matrix.loc[activity, occupation] = employment_count

    # Strategic Hierarchy for wage calculation
    for activity in activities:
        for occupation in occupations:
            strategy_used = None
            calculated_wage = np.nan

            # LEVEL 1: Employment-weighted (BEST) ✅
            employment_count = employment_matrix.loc[activity, occupation]
            if pd.notna(employment_count) and employment_count > 0:
                calculated_wage = calculate_employment_weighted_wage(
                    year_activity, year_occupation, employment_matrix, activity, occupation
                )
                strategy_used = "employment_weighted"

            # LEVEL 2: Both wages exist - Geometric mean
            elif has_activity_wage(year_activity, activity) and has_occupation_wage(year_occupation, occupation):
                activity_wage = get_activity_wage(year_activity, activity)
                occupation_wage = get_occupation_wage(year_occupation, occupation)
                if pd.notna(activity_wage) and pd.notna(occupation_wage) and activity_wage > 0 and occupation_wage > 0:
                    calculated_wage = np.sqrt(activity_wage * occupation_wage)
                    strategy_used = "geometric_mean"

            # LEVEL 3: Activity wage only
            elif has_activity_wage(year_activity, activity):
                activity_wage = get_activity_wage(year_activity, activity)
                if pd.notna(activity_wage):
                    calculated_wage = activity_wage
                    strategy_used = "activity_only"

            # LEVEL 4: Occupation wage only
            elif has_occupation_wage(year_occupation, occupation):
                occupation_wage = get_occupation_wage(year_occupation, occupation)
                if pd.notna(occupation_wage):
                    calculated_wage = occupation_wage
                    strategy_used = "occupation_only"

            # LEVEL 5: Regional imputation
            elif country in similar_countries:
                calculated_wage = impute_from_similar_countries(
                    activity, occupation, country, year, similar_countries, all_activity_df, all_occupation_df
                )
                strategy_used = "regional_imputation"

            # LEVEL 6: Global median (last resort)
            else:
                calculated_wage = calculate_global_median(
                    activity, occupation, all_activity_df, all_occupation_df
                )
                strategy_used = "global_median"

            wage_matrix.loc[activity, occupation] = calculated_wage
            strategy_log[f"{activity}_{occupation}"] = strategy_used

    return wage_matrix, employment_matrix, strategy_log


def calculate_employment_weighted_wage(year_activity, year_occupation, employment_matrix, activity, occupation):
    """Calculate employment-weighted wage for specific activity-occupation combination"""
    employment_weight = employment_matrix.loc[activity, occupation]
    total_activity_employment = employment_matrix.loc[activity, :].sum()
    total_occupation_employment = employment_matrix.loc[:, occupation].sum()

    activity_wage = get_activity_wage(year_activity, activity)
    occupation_wage = get_occupation_wage(year_occupation, occupation)

    if (pd.notna(activity_wage) and pd.notna(occupation_wage) and
            total_activity_employment > 0 and total_occupation_employment > 0):

        activity_weight = employment_weight / total_activity_employment
        occupation_weight = employment_weight / total_occupation_employment

        return (activity_wage * activity_weight + occupation_wage * occupation_weight) / (
                    activity_weight + occupation_weight)
    elif pd.notna(activity_wage) and pd.notna(occupation_wage):
        return np.sqrt(activity_wage * occupation_wage)
    elif pd.notna(activity_wage):
        return activity_wage
    elif pd.notna(occupation_wage):
        return occupation_wage
    else:
        return np.nan


def has_activity_wage(year_activity, activity):
    """Check if activity wage exists"""
    return len(year_activity[year_activity['Economic_activity'] == activity]) > 0


def has_occupation_wage(year_occupation, occupation):
    """Check if occupation wage exists"""
    return len(year_occupation[year_occupation['Occupation'] == occupation]) > 0


def get_activity_wage(year_activity, activity):
    """Get wage for specific activity"""
    match = year_activity[year_activity['Economic_activity'] == activity]
    return match['Value'].iloc[0] if len(match) > 0 else np.nan


def get_occupation_wage(year_occupation, occupation):
    """Get wage for specific occupation"""
    match = year_occupation[year_occupation['Occupation'] == occupation]
    return match['Value'].iloc[0] if len(match) > 0 else np.nan


def impute_from_similar_countries(activity, occupation, country, year, similar_countries, all_activity_df,
                                  all_occupation_df):
    """Impute wage from similar countries"""
    similar_list = similar_countries.get(country, [])

    for similar_country in similar_list:
        # Try to find wage in similar country
        similar_activity = all_activity_df[
            (all_activity_df['Country'] == similar_country) &
            (all_activity_df['Year'] == year) &
            (all_activity_df['Economic_activity'] == activity)
            ]
        similar_occupation = all_occupation_df[
            (all_occupation_df['Country'] == similar_country) &
            (all_occupation_df['Year'] == year) &
            (all_occupation_df['Occupation'] == occupation)
            ]

        if len(similar_activity) > 0 and len(similar_occupation) > 0:
            act_wage = similar_activity['Value'].iloc[0]
            occ_wage = similar_occupation['Value'].iloc[0]
            if pd.notna(act_wage) and pd.notna(occ_wage) and act_wage > 0 and occ_wage > 0:
                return np.sqrt(act_wage * occ_wage)

    # Fallback to global median
    return calculate_global_median(activity, occupation, all_activity_df, all_occupation_df)


def calculate_global_median(activity, occupation, all_activity_df, all_occupation_df):
    """Calculate global median for activity-occupation combination"""
    activity_wages = all_activity_df[all_activity_df['Economic_activity'] == activity]['Value']
    occupation_wages = all_occupation_df[all_occupation_df['Occupation'] == occupation]['Value']

    activity_wages = activity_wages.dropna()
    occupation_wages = occupation_wages.dropna()

    if len(activity_wages) > 0 and len(occupation_wages) > 0:
        return np.sqrt(activity_wages.median() * occupation_wages.median())
    elif len(activity_wages) > 0:
        return activity_wages.median()
    elif len(occupation_wages) > 0:
        return occupation_wages.median()
    else:
        # Final fallback - overall median
        return np.sqrt(all_activity_df['Value'].median() * all_occupation_df['Value'].median())


def analyze_data_quality(strategy_log):
    """Analyze data quality based on strategy usage"""
    if not strategy_log:
        return {'quality_score': 0, 'strategy_counts': {}, 'total_cells': 0}

    strategy_counts = pd.Series(list(strategy_log.values())).value_counts()

    # Quality scoring (higher weight for better strategies)
    quality_weights = {
        'employment_weighted': 1.0,
        'geometric_mean': 0.8,
        'activity_only': 0.6,
        'occupation_only': 0.6,
        'regional_imputation': 0.4,
        'global_median': 0.2
    }

    total_cells = len(strategy_log)
    quality_score = sum(
        strategy_counts.get(strategy, 0) * weight for strategy, weight in quality_weights.items()) / total_cells * 100

    return {
        'quality_score': quality_score,
        'strategy_counts': strategy_counts.to_dict(),
        'total_cells': total_cells
    }


# Run the strategic hierarchy processing
print("🚀 Starting Strategic Hierarchy Processing...")
matrices_data, quality_report = process_batch_with_strategic_hierarchy()

print(f"\n✅ Strategic Hierarchy processing complete!")
print(f"📊 Ready for employment-weighted cross-country wage analysis!")

import pandas as pd

all_rows = []

for key, data in matrices_data.items():
    country = data['country']
    year = data['year']
    wage_matrix = data['wage_matrix']
    strategy_log = data['strategy_log']
    # Get overall quality score for this country-year
    quality_score = quality_report.get(f"{country}_{year}", {}).get('quality_score', None)
    # Convert matrix to long format
    df_long = wage_matrix.reset_index().melt(
        id_vars=wage_matrix.index.name or 'index',
        var_name="Occupation",
        value_name="Wage"
    )
    # Fix activity column name if needed
    if wage_matrix.index.name:
        df_long.rename(columns={wage_matrix.index.name: "Economic_activity"}, inplace=True)
    else:
        df_long.rename(columns={'index': "Economic_activity"}, inplace=True)
    df_long["Country"] = country
    df_long["Year"] = year
    # Add per-cell strategy/robustness
    df_long["Strategy"] = df_long.apply(
        lambda row: strategy_log.get(f"{row['Economic_activity']}_{row['Occupation']}"), axis=1
    )
    # Add overall quality score
    df_long["Quality_Score"] = quality_score
    all_rows.append(df_long)

# Combine all into one DataFrame
df_all = pd.concat(all_rows, ignore_index=True)
# Reorder columns
df_all = df_all[["Country", "Year", "Economic_activity", "Occupation", "Wage", "Strategy", "Quality_Score"]]

# Save to CSV
df_all.to_csv("all_wage_matrices_with_quality_and_strategy.csv", index=False)
print("✅ Saved all wage matrices with cell strategies and overall quality score to all_wage_matrices_with_quality_and_strategy.csv")