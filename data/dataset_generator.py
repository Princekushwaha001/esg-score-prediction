import numpy as np
import pandas as pd
from typing import Tuple


class ESGDatasetGenerator:
    """Generate synthetic ESG dataset with realistic correlations."""

    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        np.random.seed(random_seed)

        self.company_prefixes = ['Green', 'Eco', 'Sustainable', 'Future', 'Clean',
                                 'Smart', 'Global', 'Innovative', 'Tech', 'Dynamic']
        self.company_suffixes = ['Corp', 'Industries', 'Solutions', 'Systems', 'Energy',
                                 'Manufacturing', 'Group', 'Holdings', 'Enterprises', 'Technologies']

    def generate(self, num_companies: int = 100) -> pd.DataFrame:
        """
        Generate synthetic ESG dataset.

        Args:
            num_companies: Number of companies to generate

        Returns:
            DataFrame with ESG data
        """
        data = []

        for i in range(num_companies):
            # Generate features with realistic ranges
            co2_emissions = np.random.randint(10000, 500000)
            energy_use = np.random.randint(5000, 200000)
            diversity_index = np.random.randint(20, 96)
            governance_rating = np.random.randint(3, 11)

            # Calculate ESG score with realistic correlations
            # Lower emissions and energy use = better score
            # Higher diversity and governance = better score
            env_score = 100 - (co2_emissions / 5000) - (energy_use / 2000)
            social_score = diversity_index * 0.8
            gov_score = governance_rating * 8

            # Weighted average with noise
            esg_score = (env_score * 0.4 + social_score * 0.3 + gov_score * 0.3 +
                         np.random.normal(0, 5))

            # Clamp to valid range
            esg_score = max(20, min(100, int(esg_score)))

            # Generate company name
            company_name = f"{np.random.choice(self.company_prefixes)} {np.random.choice(self.company_suffixes)}"

            data.append({
                'Company': company_name,
                'CO2_Emissions': co2_emissions,
                'Energy_Use': energy_use,
                'Diversity_Index': diversity_index,
                'Governance_Rating': governance_rating,
                'ESG_Score': esg_score
            })

        return pd.DataFrame(data)

    def save_to_csv(self, df: pd.DataFrame, filepath: str):
        """Save dataset to CSV file."""
        df.to_csv(filepath, index=False)
        print(f"Dataset saved to {filepath}")
