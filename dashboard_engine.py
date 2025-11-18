
import pandas as pd
from sqlalchemy import create_engine, text
from datetime import datetime
from typing import Dict, Any
import logging
from urllib.parse import quote_plus

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DashboardEngine:
    """Dashboard Engine - Executive Analytics Platform"""
    
    def __init__(self, server: str, database: str, username: str, password: str):
        """Initialize with database connection parameters"""
        self.server = server
        self.database = database
        self.username = username
        self.password = password
        self.engine = None
    
    def connect(self) -> bool:
        """Establish database connection with connection pooling"""
        try:
            conn_string = f"mssql+pymssql://{self.username}:{quote_plus(self.password)}@{self.server}/{self.database}"
            self.engine = create_engine(
                conn_string,
                pool_pre_ping=True,
                pool_recycle=3600,
                pool_size=5,
                max_overflow=10
            )
            
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            logger.info("Dashboard engine connected successfully")
            return True
        except Exception as e:
            logger.error(f"Dashboard engine connection failed: {e}")
            return False
    
    def get_date_filter(self, filter_type: str):
        """Get date filter boundaries"""
        now = pd.Timestamp.now()
        
        if filter_type == 'year':
            filter_start = now - pd.Timedelta(days=365)
            filter_name = "Last 365 Days"
        elif filter_type == 'month':
            filter_start = now - pd.Timedelta(days=30)
            filter_name = "Last 30 Days"
        elif filter_type == 'week':
            filter_start = now - pd.Timedelta(days=7)
            filter_name = "Last 7 Days"
        elif filter_type == 'ytd':
            filter_start = now.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
            filter_name = "Year To Date (2025)"
        else:
            filter_start = pd.Timestamp('2020-01-01')
            filter_name = "All Time"
        
        return filter_start, filter_name, now
    
    def calc_metrics(self, leads, meetings, start, end, won_stage_id=7):
        """
        Calculate KPI metrics for a time period
        ✅ UPDATED: Now counts won deals by LeadStageId = 7 (Converted stage)
        """
        period_leads = leads[(leads['CreatedOn'] >= start) & (leads['CreatedOn'] <= end)].copy()
        period_meetings = meetings[(meetings['StartDateTime'] >= start) & (meetings['StartDateTime'] <= end)].copy()
        
        if 'MeetingStatusId' in period_meetings.columns:
            period_meetings = period_meetings[period_meetings['MeetingStatusId'].isin([1, 6])]
        
        total = len(period_leads)
        
        # ✅ CHANGED: Count won deals by LeadStageId = 7 instead of LeadStatusId
        if 'LeadStageId' in period_leads.columns:
            won = int(period_leads['LeadStageId'].eq(won_stage_id).sum()) if total > 0 else 0
        else:
            won = 0
        
        conv_pct = round((won / total * 100.0) if total > 0 else 0.0, 1)
        meetings_count = int(period_meetings['LeadId'].nunique()) if 'LeadId' in period_meetings.columns and len(period_meetings) > 0 else 0
        
        return {
            'total_leads': total,
            'conversion_rate': conv_pct,
            'meetings_scheduled': meetings_count,
            'won_deals': won
        }
    
    def get_complete_dashboard(self, date_filter: str = 'year') -> Dict[str, Any]:
        """Get complete dashboard data - MATCHES STREAMLIT LOGIC EXACTLY"""
        try:
            if not self.connect():
                return {'status': 'failed', 'error': 'Database connection failed'}
            
            logger.info(f"Generating dashboard with filter: {date_filter}")
            
            # Load data
            leads_all = pd.read_sql("SELECT * FROM dbo.Lead WHERE IsActive = 1", self.engine)
            leads_all['CreatedOn'] = pd.to_datetime(leads_all['CreatedOn'])
            
            meetings_all = pd.read_sql("""
                SELECT * FROM dbo.AgentMeetingAssignment 
                WHERE StartDateTime >= DATEADD(MONTH, -12, GETDATE())
            """, self.engine)
            meetings_all['StartDateTime'] = pd.to_datetime(meetings_all['StartDateTime'])
            
            statuses = pd.read_sql("SELECT * FROM dbo.LeadStatus WHERE IsActive = 1", self.engine)
            stages = pd.read_sql("SELECT * FROM dbo.LeadStage WHERE IsActive = 1", self.engine)
            countries = pd.read_sql("SELECT * FROM dbo.Country", self.engine)
            audit = pd.read_sql("SELECT * FROM dbo.LeadStageAudit", self.engine)
            
            # ✅ CHANGED: Use LeadStageId = 7 for won deals (Converted stage)
            won_stage_id = 7
            
            # Apply date filter
            filter_start, filter_name, now = self.get_date_filter(date_filter)
            leads_filtered = leads_all[leads_all['CreatedOn'] >= filter_start].copy()
            
            # Calculate KPIs
            week_start = now - pd.Timedelta(days=now.weekday())
            week_start = week_start.replace(hour=0, minute=0, second=0, microsecond=0)
            month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            year_start = now.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
            
            # ✅ Pass LeadStageId = 7 instead of LeadStatusId
            wtd = self.calc_metrics(leads_filtered, meetings_all, week_start, now, won_stage_id)
            mtd = self.calc_metrics(leads_filtered, meetings_all, month_start, now, won_stage_id)
            ytd = self.calc_metrics(leads_filtered, meetings_all, year_start, now, won_stage_id)
            
            # Performance Trends
            leads_filtered['YearMonth'] = leads_filtered['CreatedOn'].dt.to_period('M')
            
            # ✅ CHANGED: Count by LeadStageId = 7 instead of LeadStatusId
            monthly = leads_filtered.groupby('YearMonth').agg({
                'LeadId': 'count',
                'LeadStageId': lambda x: (x == 7).sum()  # ✅ Count by LeadStageId = 7
            }).reset_index()
            monthly.columns = ['Month', 'Total', 'Won']
            monthly['Month'] = monthly['Month'].astype(str)
            monthly['ConvRate'] = (monthly['Won'] / monthly['Total'] * 100).fillna(0).round(1)
            monthly_last6 = monthly.tail(6)
            
            meetings_all['YearMonth'] = meetings_all['StartDateTime'].dt.to_period('M')
            monthly_meetings = meetings_all.groupby('YearMonth')['LeadId'].nunique().reset_index()
            monthly_meetings.columns = ['Month', 'Meetings']
            monthly_meetings['Month'] = monthly_meetings['Month'].astype(str)
            
            trends_data = []
            for idx, row in monthly_last6.iterrows():
                meetings_count = 0
                meeting_row = monthly_meetings[monthly_meetings['Month'] == row['Month']]
                if not meeting_row.empty:
                    meetings_count = int(meeting_row.iloc[0]['Meetings'])
                
                trends_data.append({
                    'month': row['Month'],
                    'total_leads': int(row['Total']),
                    'won_deals': int(row['Won']),
                    'conversion_rate': float(row['ConvRate']),
                    'meetings': meetings_count
                })
            
            # ============================================================================
            # FUNNEL - EXACT STREAMLIT LOGIC
            # ============================================================================
            
            # ✅ Filter for active leads only (matching Streamlit)
            active_leads = leads_filtered[leads_filtered.get("IsActive", 1) == 1].copy()
            total_leads = len(active_leads)
            
            if not audit.empty and not stages.empty and "StageId" in audit.columns:
                # ✅ EXACT STREAMLIT LOGIC: Merge audit with stages, then filter by active leads
                funnel_query = audit.merge(
                    stages[["LeadStageId", "StageName_E", "SortOrder"]],
                    left_on="StageId",
                    right_on="LeadStageId",
                    how="inner"
                ).merge(
                    active_leads[["LeadId"]],  # ✅ Only keep active leads
                    on="LeadId",
                    how="inner"  # ✅ Inner join filters out inactive leads
                )
                
                funnel_df = (
                    funnel_query.groupby(["SortOrder", "StageName_E"], as_index=False)["LeadId"]
                    .nunique()
                    .rename(columns={"LeadId": "Count"})
                    .sort_values("SortOrder", ascending=True)
                )
                
                stage_rename = {
                    "New": "New Leads",
                    "Qualified": "Qualified",
                    "Followup Process": "Follow-up",
                    "Meeting Scheduled": "Meetings",
                    "Negotiation": "Under Negotiation",
                    "Won": "Converted"
                }
                
                funnel_df["Stage"] = funnel_df["StageName_E"].map(stage_rename).fillna(funnel_df["StageName_E"])
                funnel_df = funnel_df[funnel_df["Stage"] != "Lost"].reset_index(drop=True)
            else:
                funnel_df = pd.DataFrame([{"Stage": "Total Leads", "Count": total_leads, "SortOrder": 0}])
            
            # ✅ Calculate ONLY percentage from initial (matching Plotly's "percent initial")
            funnel_data = []
            initial_count = funnel_df.iloc[0]['Count'] if len(funnel_df) > 0 else 1
            
            for idx, row in funnel_df.iterrows():
                count = int(row['Count'])
                # Percentage from initial/top stage (matches Plotly "percent initial")
                percentage = round((count / initial_count * 100.0) if initial_count > 0 else 0.0, 1)
                
                funnel_data.append({
                    'stage': row['Stage'],
                    'count': count,
                    'percentage': percentage,
                    'sort_order': int(row['SortOrder'])
                })
            
            # Overall conversion
            converted_count = 0
            for stage_data in funnel_data:
                if stage_data['stage'] in ['Converted', 'Won']:
                    converted_count = stage_data['count']
                    break
            
            overall_conversion_rate = round((converted_count / initial_count * 100.0) if initial_count > 0 else 0.0, 1)
            
            # ============================================================================
            # TOP MARKETS (Top 10 Countries) - Using active leads only
            # ============================================================================
            
            market_analysis = active_leads.copy()
            market_analysis['CountryId'] = market_analysis['CountryId'].fillna(-1)
            
            market_summary = (
                market_analysis.groupby("CountryId", as_index=False)
                .size()
                .rename(columns={"size": "Leads"})
            )
            
            market_summary = market_summary.merge(
                countries[["CountryId", "CountryName_E"]],
                on="CountryId",
                how="left"
            )
            
            market_summary['CountryName_E'] = market_summary['CountryName_E'].fillna('Unknown/Not Set')
            
            total_leads_in_markets = market_summary["Leads"].sum()
            market_summary["Share"] = (market_summary["Leads"] / total_leads_in_markets * 100.0).round(1)
            top_markets = market_summary.nlargest(10, "Leads")
            
            markets_data = [
                {'country': row['CountryName_E'], 'leads': int(row['Leads']), 'share': float(row['Share'])}
                for idx, row in top_markets.iterrows()
            ]
            
            # Build response
            response = {
                'status': 'success',
                'date_filter': date_filter,
                'filter_name': filter_name,
                'total_active_leads': len(leads_all),
                'filtered_leads': len(leads_filtered),
                'kpis': {'wtd': wtd, 'mtd': mtd, 'ytd': ytd},
                'trends': trends_data,
                'funnel': funnel_data,
                'total_leads': total_leads,
                'total_converted': converted_count,
                'overall_conversion_rate': overall_conversion_rate,
                'top_markets': markets_data,
                'timestamp': datetime.now().isoformat()
            }
            
            self.engine.dispose()
            logger.info("Dashboard data generated successfully")
            return response
            
        except Exception as e:
            logger.error(f"Dashboard generation failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def get_executive_summary(self, date_filter: str = 'year') -> Dict[str, Any]:
        """Get executive summary"""
        return self.get_complete_dashboard(date_filter)
