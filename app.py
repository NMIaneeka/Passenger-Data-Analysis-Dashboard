import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# OOP Classes for Analysis
# =========================

class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None

    def load_data(self):
        self.df = pd.read_csv(self.file_path)
        self.df['Timestamp'] = pd.to_datetime(self.df['Timestamp'])
        self.df['Departure_Time'] = pd.to_datetime(self.df['Departure_Time'], errors='coerce')
        self.df['Arrival_Time'] = pd.to_datetime(self.df['Arrival_Time'], errors='coerce')
        return self.df


class PeakCongestionAnalysis:
    def __init__(self, df):
        self.df = df

    def hourly_analysis(self):
        return self.df.groupby(self.df['Timestamp'].dt.hour)['passenger_count'].sum()

    def daily_analysis(self):
        return self.df.groupby(self.df['Timestamp'].dt.day)['passenger_count'].sum()

    def monthly_analysis(self):
        return self.df.groupby(self.df['Timestamp'].dt.month)['passenger_count'].sum()

    def yearly_analysis(self):
        return self.df.groupby(self.df['Timestamp'].dt.year)['passenger_count'].sum()


class PopularRouteAnalysis:
    def __init__(self, df):
        self.df = df

    def top_routes(self, n=5):
        return self.df.groupby('Route_ID')['passenger_count'].sum().sort_values(ascending=False).head(n)

    def top_transfer_points(self, n=5):
        transfers = pd.concat([self.df['Entry_Station_ID'], self.df['Exit_Station_ID']])
        return transfers.value_counts().head(n)


class ServiceDisruptionDetection:
    def __init__(self, df):
        self.df = df

    def detect_anomalies(self, z_thresh=2.5):
        daily_counts = self.df.groupby(self.df['Timestamp'].dt.date)['passenger_count'].sum()
        mean, std = daily_counts.mean(), daily_counts.std()
        anomalies = daily_counts[(daily_counts - mean).abs() > z_thresh * std]
        return anomalies, daily_counts


class RegionalPerformanceAnalysis:
    def __init__(self, df):
        self.df = df

    def region_passenger_trends(self):
        return self.df.groupby('Region')['passenger_count'].sum().sort_values(ascending=False)

    def region_revenue_trends(self):
        return self.df.groupby('Region')['Total_Fees'].sum().sort_values(ascending=False)


# =========================
# UI Classes
# =========================

class FilterUI:
    @staticmethod
    def apply_filters(df):
        st.sidebar.subheader("Filters")
        regions = st.sidebar.multiselect("Select Regions", df['Region'].unique())
        if regions:
            df = df[df['Region'].isin(regions)]

        routes = st.sidebar.multiselect("Select Routes", df['Route_ID'].unique())
        if routes:
            df = df[df['Route_ID'].isin(routes)]
        return df


class PeakCongestionUI:
    def __init__(self, df):
        self.analysis = PeakCongestionAnalysis(df)

    def render(self):
        st.header("1. Peak Congestion Analysis")

        # Let user choose granularity
        view_type = st.radio(
            "Choose time granularity for congestion view:",
            ("Hourly", "Daily", "Monthly", "Yearly"),
            horizontal=True
        )

        fig, ax = plt.subplots()

        if view_type == "Hourly":
            self.analysis.hourly_analysis().plot(ax=ax, marker="o", color="blue")
            ax.set_title("Hourly Passenger Congestion")
            ax.set_xlabel("Hour of Day")
        elif view_type == "Daily":
            self.analysis.daily_analysis().plot(ax=ax, marker="s", color="green")
            ax.set_title("Daily Passenger Congestion")
            ax.set_xlabel("Day of Month")
        elif view_type == "Monthly":
            self.analysis.monthly_analysis().plot(ax=ax, marker="^", color="orange")
            ax.set_title("Monthly Passenger Congestion")
            ax.set_xlabel("Month")
        else:  # Yearly
            self.analysis.yearly_analysis().plot(ax=ax, marker="x", color="red")
            ax.set_title("Yearly Passenger Congestion")
            ax.set_xlabel("Year")

        ax.set_ylabel("Passenger Count")
        ax.grid(True, linestyle="--", alpha=0.6)

        st.pyplot(fig)



class PopularRouteUI:
    def __init__(self, df):
        self.analysis = PopularRouteAnalysis(df)

    def render(self):
        st.header("2. Popular Routes & Transfer Points")

        # Horizontal bar chart
        fig, ax = plt.subplots()
        self.analysis.top_routes().plot(kind="barh", ax=ax, color="skyblue")
        ax.set_title("Top Routes by Passenger Count")
        ax.set_xlabel("Passengers")
        ax.set_ylabel("Route ID")
        st.pyplot(fig)

        fig, ax = plt.subplots()
        self.analysis.top_transfer_points().plot(kind="barh", ax=ax, color="lightgreen")
        ax.set_title("Top Transfer Points")
        ax.set_xlabel("Transfers")
        ax.set_ylabel("Station ID")
        st.pyplot(fig)


class ServiceDisruptionUI:
    def __init__(self, df):
        self.analysis = ServiceDisruptionDetection(df)

    def render(self):
        st.header("3. Service Disruption Detection")
        anomalies, daily_counts = self.analysis.detect_anomalies()

        st.write("Anomalies Detected:")
        st.dataframe(anomalies)

        # Line chart with anomaly markers
        fig, ax = plt.subplots()
        daily_counts.plot(ax=ax, label="Daily Counts", color="blue")
        ax.scatter(anomalies.index, anomalies.values, color="red", label="Anomalies", zorder=5)
        ax.set_title("Passenger Count with Anomalies")
        ax.set_ylabel("Passengers")
        ax.set_xlabel("Date")
        ax.legend()
        st.pyplot(fig)


class RegionalPerformanceUI:
    def __init__(self, df):
        self.analysis = RegionalPerformanceAnalysis(df)

    def render(self):
        st.header("4. Regional Performance Analysis")

        # Pie chart for passenger trends
        fig, ax = plt.subplots()
        self.analysis.region_passenger_trends().plot(kind="pie", ax=ax, autopct="%1.1f%%")
        ax.set_ylabel("")
        ax.set_title("Passenger Distribution by Region")
        st.pyplot(fig)

        # Pie chart for revenue trends
        fig, ax = plt.subplots()
        self.analysis.region_revenue_trends().plot(kind="pie", ax=ax, autopct="%1.1f%%")
        ax.set_ylabel("")
        ax.set_title("Revenue Distribution by Region")
        st.pyplot(fig)


# =========================
# Main App Controller
# =========================

class PassengerDashboardApp:
    def __init__(self):
        self.df = None

    def run(self):
        st.title("🚇 Passenger Data Analysis Dashboard")
        st.sidebar.header("Upload Data")

        uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

        if uploaded_file:
            self.df = DataLoader(uploaded_file).load_data()
            self.df = FilterUI.apply_filters(self.df)

            st.write("### Data Preview")
            st.dataframe(self.df.head())

            PeakCongestionUI(self.df).render()
            PopularRouteUI(self.df).render()
            ServiceDisruptionUI(self.df).render()
            RegionalPerformanceUI(self.df).render()
        else:
            st.info("👆 Please upload a dataset CSV file to proceed.")


# =========================
# Run the App
# =========================
if __name__ == "__main__":
    PassengerDashboardApp().run()
