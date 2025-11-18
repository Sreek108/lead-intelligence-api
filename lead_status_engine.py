import pandas as pd
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import logging
from urllib.parse import quote_plus


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LeadStatusEngine:
    """Complete Lead Status Analytics Engine with all endpoints"""
    
    def __init__(self, server: str, database: str, username: str, password: str):
        """Initialize with database credentials"""
        self.server = server
        self.database = database
        self.username = username
        self.password = password
        self.engine = None
        self.scoring_master = None
    
    def connect(self) -> bool:
        """Establish database connection"""
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
            
            logger.info("✅ Lead Status Engine: Connected")
            return True
        except Exception as e:
            logger.error(f"❌ Connection failed: {e}")
            return False
    
    def _load_scoring_master(self) -> pd.DataFrame:
        """Load LeadScoring master table"""
        try:
            query = """
            SELECT LeadScoringId, ScoreName_E as ScoreName, ScoreRange, MinScore, MaxScore, IsActive
            FROM dbo.LeadScoring
            WHERE IsActive = 1
            ORDER BY LeadScoringId
            """
            self.scoring_master = pd.read_sql(query, self.engine)
            return self.scoring_master
        except Exception as e:
            logger.error(f"❌ Failed to load scoring master: {e}")
            raise
    
    def _load_data(self) -> pd.DataFrame:
        """Load lead data"""
        try:
            query = """
            SELECT 
                L.LeadId, L.LeadCode, L.LeadScoringId,
                LS_Scoring.ScoreName_E as ScoringCategory,
                L.LeadStageId, L.LeadStatusId,
                LS_Status.StatusName_E as Status,
                L.CreatedOn, L.ModifiedOn, L.IsActive
            FROM dbo.Lead L
            LEFT JOIN dbo.LeadScoring LS_Scoring ON L.LeadScoringId = LS_Scoring.LeadScoringId
            LEFT JOIN dbo.LeadStatus LS_Status ON L.LeadStatusId = LS_Status.LeadStatusId
            WHERE L.IsActive = 1
            """
            leads = pd.read_sql(query, self.engine)
            leads['CreatedOn'] = pd.to_datetime(leads['CreatedOn'])
            leads['ModifiedOn'] = pd.to_datetime(leads['ModifiedOn'])
            leads['Status'] = leads['Status'].fillna('Not Assigned')
            leads['ScoringCategory'] = leads['ScoringCategory'].fillna('Not Scored')
            return leads
        except Exception as e:
            logger.error(f"❌ Data loading failed: {e}")
            raise
    
    def _categorize_leads_dynamic(self, L: pd.DataFrame) -> Dict[str, Any]:
        """Categorize leads dynamically"""
        results = {'by_scoring': {}, 'won': 0, 'other': 0}
        
        for idx, score_row in self.scoring_master.iterrows():
            scoring_id = score_row['LeadScoringId']
            score_name = score_row['ScoreName']
            count = int((L['LeadScoringId'] == scoring_id).sum())
            results['by_scoring'][score_name] = {
                'scoring_id': int(scoring_id),
                'count': count,
                'score_range': score_row['ScoreRange']
            }
        
        results['won'] = int((L['LeadStageId'] == 7).sum())
        results['other'] = int(L['LeadScoringId'].isna().sum())
        return results
    
    def _calculate_metrics(self, L: pd.DataFrame, won_count: int) -> tuple:
        """Calculate conversion metrics"""
        total = len(L)
        conversion_rate = round((won_count / total * 100), 1) if total > 0 else 0.0
        
        converted_leads = L[L['LeadStageId'] == 7].copy()
        if not converted_leads.empty:
            converted_leads['DaysToClose'] = (datetime.now() - converted_leads['CreatedOn']).dt.days
            avg_days = round(converted_leads['DaysToClose'].mean(), 1)
        else:
            avg_days = 0.0
        
        return conversion_rate, avg_days
    
    # ========================================================================
    # PUBLIC API METHODS
    # ========================================================================
    
    def get_complete_analytics(self) -> Dict[str, Any]:
        """Get complete analytics (main endpoint)"""
        try:
            if not self.connect():
                return {'status': 'failed', 'error': 'Database connection failed'}
            
            self._load_scoring_master()
            L = self._load_data()
            
            if L.empty:
                return {'status': 'failed', 'error': 'No active leads found'}
            
            total_leads = len(L)
            category_results = self._categorize_leads_dynamic(L)
            conversion_rate, avg_days = self._calculate_metrics(L, category_results['won'])
            
            overview = {
                'total_leads': total_leads,
                'won_leads': category_results['won'],
                'not_scored_leads': category_results['other'],
                'conversion_rate': conversion_rate,
                'avg_days_to_close': avg_days
            }
            
            for score_name, score_data in category_results['by_scoring'].items():
                overview[f"{score_name.lower()}_leads"] = score_data['count']
            
            distribution = []
            for score_name, score_data in category_results['by_scoring'].items():
                distribution.append({
                    'category': score_name,
                    'scoring_id': score_data['scoring_id'],
                    'count': score_data['count'],
                    'percentage': round(score_data['count'] / total_leads * 100, 1) if total_leads > 0 else 0,
                    'score_range': score_data['score_range']
                })
            
            distribution.append({
                'category': 'Not Scored',
                'scoring_id': None,
                'count': category_results['other'],
                'percentage': round(category_results['other'] / total_leads * 100, 1) if total_leads > 0 else 0,
                'score_range': 'N/A'
            })
            
            response = {
                'status': 'success',
                'version': '5.4-production',
                'timestamp': datetime.now().isoformat(),
                'overview': overview,
                'distribution': distribution,
                'status_distribution': self._get_status_distribution_internal(L),
                'top_3_statuses': self._get_top_statuses_internal(L)
            }
            
            self.engine.dispose()
            return response
            
        except Exception as e:
            logger.error(f"❌ Analysis failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def get_overview_only(self) -> Dict[str, Any]:
        """Get overview metrics only"""
        try:
            if not self.connect():
                return {'status': 'failed', 'error': 'Database connection failed'}
            
            self._load_scoring_master()
            L = self._load_data()
            
            if L.empty:
                return {'status': 'failed', 'error': 'No active leads found'}
            
            total_leads = len(L)
            category_results = self._categorize_leads_dynamic(L)
            conversion_rate, avg_days = self._calculate_metrics(L, category_results['won'])
            
            overview = {
                'total_leads': total_leads,
                'won_leads': category_results['won'],
                'not_scored_leads': category_results['other'],
                'conversion_rate': conversion_rate,
                'avg_days_to_close': avg_days
            }
            
            for score_name, score_data in category_results['by_scoring'].items():
                overview[f"{score_name.lower()}_leads"] = score_data['count']
            
            self.engine.dispose()
            return {'status': 'success', 'version': '5.4', 'timestamp': datetime.now().isoformat(), 'overview': overview}
            
        except Exception as e:
            logger.error(f"❌ Failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def get_status_distribution(self) -> Dict[str, Any]:
        """Get lead distribution by status"""
        try:
            if not self.connect():
                return {'status': 'failed', 'error': 'Database connection failed'}
            
            L = self._load_data()
            
            if L.empty:
                return {'status': 'failed', 'error': 'No active leads found'}
            
            status_counts = L['Status'].value_counts().reset_index()
            status_counts.columns = ['Status', 'Count']
            
            total_leads = len(L)
            
            status_distribution = []
            for idx, row in status_counts.iterrows():
                count = int(row['Count'])
                percentage = round((count / total_leads * 100), 1) if total_leads > 0 else 0.0
                status_distribution.append({
                    'status': row['Status'],
                    'count': count,
                    'percentage': percentage
                })
            
            top_3_statuses = []
            for idx, row in status_counts.head(3).iterrows():
                top_3_statuses.append({
                    'Status': row['Status'],
                    'Count': int(row['Count'])
                })
            
            self.engine.dispose()
            return {
                'status': 'success',
                'status_distribution': status_distribution,
                'top_3_statuses': top_3_statuses,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def get_status_comparison(self) -> Dict[str, Any]:
        """Get comprehensive status comparison matrix"""
        try:
            if not self.connect():
                return {'status': 'failed', 'error': 'Database connection failed'}
            
            L = self._load_data()
            
            if L.empty:
                return {'status': 'failed', 'error': 'No active leads found'}
            
            now = datetime.now()
            L['Age_Days'] = (now - L['CreatedOn']).dt.days
            
            comparison_matrix = []
            
            for status in L['Status'].unique():
                status_leads = L[L['Status'] == status]
                
                total_leads = len(status_leads)
                avg_age = int(status_leads['Age_Days'].mean()) if total_leads > 0 else 0
                won_count = int((status_leads['LeadStageId'] == 7).sum())
                win_rate = round((won_count / total_leads * 100), 1) if total_leads > 0 else 0.0
                
                meetings = int(status_leads['Status'].str.contains('Meeting', case=False, na=False).sum())
                
                health_factors = []
                health_factors.append(min(40, win_rate * 0.4))
                
                max_leads = len(L)
                volume_score = (total_leads / max_leads * 30) if max_leads > 0 else 0
                health_factors.append(volume_score)
                
                if avg_age > 0:
                    freshness_score = max(0, 30 - (avg_age / 5))
                else:
                    freshness_score = 30
                health_factors.append(freshness_score)
                
                health_score = int(sum(health_factors))
                health_score = min(100, max(0, health_score))
                
                comparison_matrix.append({
                    'Status': status,
                    'Total_Leads': total_leads,
                    'Avg_Age': avg_age,
                    'Won_Count': won_count,
                    'Win_Rate': win_rate,
                    'Meetings': meetings,
                    'Health_Score': health_score
                })
            
            comparison_matrix.sort(key=lambda x: x['Total_Leads'], reverse=True)
            
            self.engine.dispose()
            return {
                'status': 'success',
                'status_comparison_matrix': comparison_matrix,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def get_trends(self, months: Optional[int] = 12) -> Dict[str, Any]:
        """Get status trends by month"""
        try:
            if not self.connect():
                return {'status': 'failed', 'error': 'Database connection failed'}
            
            L = self._load_data()
            
            if L.empty:
                return {'status': 'failed', 'error': 'No active leads found'}
            
            L['YearMonth'] = L['CreatedOn'].dt.to_period('M').astype(str)
            unique_months = sorted(L['YearMonth'].unique())
            
            trends = []
            
            for month in unique_months:
                month_leads = L[L['YearMonth'] == month]
                status_counts = month_leads['Status'].value_counts()
                
                statuses = []
                for status, count in status_counts.items():
                    statuses.append({
                        'Status': status,
                        'Count': int(count)
                    })
                
                if statuses:
                    trends.append({
                        'month': month,
                        'statuses': statuses
                    })
            
            self.engine.dispose()
            return {
                'status': 'success',
                'trends': trends,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def get_agent_performance(self) -> Dict[str, Any]:
        """Get agent performance metrics"""
        try:
            if not self.connect():
                return {'status': 'failed', 'error': 'Database connection failed'}
            
            try:
                query = """
                SELECT 
                    U.UserId,
                    U.UserName,
                    COUNT(DISTINCT L.LeadId) as AssignedLeads,
                    COUNT(DISTINCT CASE WHEN L.LeadStageId = 7 THEN L.LeadId END) as WonLeads
                FROM dbo.Lead L
                LEFT JOIN dbo.[User] U ON L.AssignedTo = U.UserId
                WHERE L.IsActive = 1 AND U.UserId IS NOT NULL
                GROUP BY U.UserId, U.UserName
                HAVING COUNT(DISTINCT L.LeadId) > 0
                ORDER BY AssignedLeads DESC
                """
                
                agents = pd.read_sql(query, self.engine)
                
                if agents.empty:
                    self.engine.dispose()
                    return {
                        'status': 'success',
                        'version': '5.4',
                        'timestamp': datetime.now().isoformat(),
                        'performance': [],
                        'message': 'No agent assignments found'
                    }
                
                agents['ConversionRate'] = ((agents['WonLeads'] / agents['AssignedLeads']) * 100).round(1)
                agents['ConversionRate'] = agents['ConversionRate'].fillna(0.0)
                
                performance = agents.to_dict('records')
                
                for agent in performance:
                    agent['AssignedLeads'] = int(agent['AssignedLeads'])
                    agent['WonLeads'] = int(agent['WonLeads'])
                    agent['ConversionRate'] = float(agent['ConversionRate'])
                
                self.engine.dispose()
                return {
                    'status': 'success',
                    'version': '5.4',
                    'timestamp': datetime.now().isoformat(),
                    'performance': performance,
                    'total_agents': len(performance)
                }
                
            except Exception as query_error:
                self.engine.dispose()
                return {
                    'status': 'success',
                    'version': '5.4',
                    'timestamp': datetime.now().isoformat(),
                    'performance': [],
                    'message': 'Agent assignment column (AssignedTo) not found in Lead table'
                }
            
        except Exception as e:
            logger.error(f"❌ Failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def get_lead_ids_drilldown(self, filter_type: str, filter_value: Optional[str] = None) -> Dict[str, Any]:
        
        try:
            if not self.connect():
                return {'status': 'failed', 'error': 'Database connection failed'}
            
            L = self._load_data()
            
            if L.empty:
                return {'status': 'failed', 'error': 'No active leads found'}
            
            now = datetime.now()
            filtered_leads = None
            
            if filter_type == 'total_leads':
                filtered_leads = L.copy()
            
            elif filter_type == 'wtd':
                week_start = now - timedelta(days=now.weekday())
                week_start = week_start.replace(hour=0, minute=0, second=0, microsecond=0)
                filtered_leads = L[L['CreatedOn'] >= week_start]
            
            elif filter_type == 'mtd':
                month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
                filtered_leads = L[L['CreatedOn'] >= month_start]
            
            elif filter_type == 'ytd':
                year_start = now.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
                filtered_leads = L[L['CreatedOn'] >= year_start]
            
            elif filter_type == 'status':
                if not filter_value:
                    return {'status': 'failed', 'error': 'filter_value required for status filter'}
                filtered_leads = L[L['Status'] == filter_value]
            
            elif filter_type == 'scoring':
                if not filter_value:
                    return {'status': 'failed', 'error': 'filter_value required for scoring filter'}
                
                self._load_scoring_master()
                
                scoring_map = {}
                for idx, row in self.scoring_master.iterrows():
                    scoring_map[row['ScoreName'].lower()] = row['LeadScoringId']
                
                filter_key = filter_value.lower().replace('_leads', '')
                if filter_key in scoring_map:
                    filtered_leads = L[L['LeadScoringId'] == scoring_map[filter_key]]
                else:
                    return {'status': 'failed', 'error': f'Invalid scoring category: {filter_value}'}
            
            elif filter_type == 'won':
                filtered_leads = L[L['LeadStageId'] == 7]
            
            elif filter_type == 'not_scored':
                filtered_leads = L[L['LeadScoringId'].isna()]
            
            elif filter_type == 'agent':
                if not filter_value:
                    return {'status': 'failed', 'error': 'filter_value required for agent filter'}
                
                query = f"""
                SELECT L.LeadId, L.LeadCode, L.LeadScoringId,
                       LS_Scoring.ScoreName_E as ScoringCategory,
                       L.LeadStageId, L.LeadStatusId,
                       LS_Status.StatusName_E as Status,
                       L.CreatedOn, L.ModifiedOn, L.IsActive, L.AssignedTo
                FROM dbo.Lead L
                LEFT JOIN dbo.LeadScoring LS_Scoring ON L.LeadScoringId = LS_Scoring.LeadScoringId
                LEFT JOIN dbo.LeadStatus LS_Status ON L.LeadStatusId = LS_Status.LeadStatusId
                WHERE L.IsActive = 1 AND L.AssignedTo = {int(filter_value)}
                """
                filtered_leads = pd.read_sql(query, self.engine)
                
                if not filtered_leads.empty:
                    filtered_leads['CreatedOn'] = pd.to_datetime(filtered_leads['CreatedOn'])
                    filtered_leads['ModifiedOn'] = pd.to_datetime(filtered_leads['ModifiedOn'])
                    filtered_leads['Status'] = filtered_leads['Status'].fillna('Not Assigned')
            
            elif filter_type == 'country':
                if not filter_value:
                    return {'status': 'failed', 'error': 'filter_value required for country filter'}
                
                query = f"""
                SELECT L.LeadId, L.LeadCode, L.LeadScoringId,
                       LS_Scoring.ScoreName_E as ScoringCategory,
                       L.LeadStageId, L.LeadStatusId,
                       LS_Status.StatusName_E as Status,
                       L.CreatedOn, L.ModifiedOn, L.IsActive,
                       C.CountryName_E as CountryName
                FROM dbo.Lead L
                LEFT JOIN dbo.LeadScoring LS_Scoring ON L.LeadScoringId = LS_Scoring.LeadScoringId
                LEFT JOIN dbo.LeadStatus LS_Status ON L.LeadStatusId = LS_Status.LeadStatusId
                LEFT JOIN dbo.Country C ON L.CountryId = C.CountryId
                WHERE L.IsActive = 1 AND C.CountryName_E = '{filter_value}'
                """
                filtered_leads = pd.read_sql(query, self.engine)
                
                if not filtered_leads.empty:
                    filtered_leads['CreatedOn'] = pd.to_datetime(filtered_leads['CreatedOn'])
                    filtered_leads['ModifiedOn'] = pd.to_datetime(filtered_leads['ModifiedOn'])
                    filtered_leads['Status'] = filtered_leads['Status'].fillna('Not Assigned')
                    filtered_leads['ScoringCategory'] = filtered_leads['ScoringCategory'].fillna('Not Scored')
            
            elif filter_type == 'all_countries':
                # Get all countries with their leads grouped
                query = """
                SELECT 
                    L.LeadId, 
                    L.LeadCode,
                    C.CountryName_E as CountryName,
                    L.LeadStageId,
                    LS_Status.StatusName_E as Status,
                    LS_Scoring.ScoreName_E as ScoringCategory,
                    L.CreatedOn,
                    L.ModifiedOn
                FROM dbo.Lead L
                LEFT JOIN dbo.Country C ON L.CountryId = C.CountryId
                LEFT JOIN dbo.LeadStatus LS_Status ON L.LeadStatusId = LS_Status.LeadStatusId
                LEFT JOIN dbo.LeadScoring LS_Scoring ON L.LeadScoringId = LS_Scoring.LeadScoringId
                WHERE L.IsActive = 1
                ORDER BY C.CountryName_E
                """
                
                leads_df = pd.read_sql(query, self.engine)
                
                if leads_df.empty:
                    self.engine.dispose()
                    return {
                        'status': 'success',
                        'filter_type': 'all_countries',
                        'timestamp': datetime.now().isoformat(),
                        'total_countries': 0,
                        'total_leads': 0,
                        'countries': {}
                    }
                
                # Handle leads without country
                leads_df['CountryName'] = leads_df['CountryName'].fillna('Unknown/Not Assigned')
                leads_df['CreatedOn'] = pd.to_datetime(leads_df['CreatedOn'])
                leads_df['ModifiedOn'] = pd.to_datetime(leads_df['ModifiedOn'])
                
                # Group by country
                countries_data = {}
                
                for country_name in leads_df['CountryName'].unique():
                    country_leads = leads_df[leads_df['CountryName'] == country_name]
                    
                    lead_ids = country_leads['LeadId'].tolist()
                    lead_details = []
                    
                    for idx, row in country_leads.iterrows():
                        lead_details.append({
                            'LeadId': int(row['LeadId']),
                            'LeadCode': str(row['LeadCode']),
                            'Status': str(row['Status']) if pd.notna(row['Status']) else 'Not Assigned',
                            'ScoringCategory': str(row['ScoringCategory']) if pd.notna(row['ScoringCategory']) else 'Not Scored',
                            'LeadStageId': int(row['LeadStageId']) if pd.notna(row['LeadStageId']) else None,
                            'CreatedOn': row['CreatedOn'].strftime('%Y-%m-%d') if pd.notna(row['CreatedOn']) else None
                        })
                    
                    countries_data[country_name] = {
                        'total_leads': len(lead_ids),
                        'lead_ids': lead_ids,
                        'lead_details': lead_details,
                        'won_leads': int((country_leads['LeadStageId'] == 7).sum())
                    }
                
                # Sort countries by lead count (descending)
                countries_data = dict(sorted(
                    countries_data.items(), 
                    key=lambda x: x[1]['total_leads'], 
                    reverse=True
                ))
                
                self.engine.dispose()
                return {
                    'status': 'success',
                    'filter_type': 'all_countries',
                    'timestamp': datetime.now().isoformat(),
                    'total_countries': len(countries_data),
                    'total_leads': len(leads_df),
                    'countries': countries_data
                }
            
            else:
                return {'status': 'failed', 'error': f'Invalid filter_type: {filter_type}'}
            
            if filtered_leads is None or filtered_leads.empty:
                return {
                    'status': 'success',
                    'filter_type': filter_type,
                    'filter_value': filter_value,
                    'total_count': 0,
                    'lead_ids': [],
                    'lead_details': [],
                    'timestamp': datetime.now().isoformat()
                }
            
            lead_ids = filtered_leads['LeadId'].tolist()
            
            lead_details = []
            for idx, row in filtered_leads.iterrows():
                detail = {
                    'LeadId': int(row['LeadId']),
                    'LeadCode': str(row['LeadCode']),
                    'Status': str(row['Status']),
                    'ScoringCategory': str(row['ScoringCategory']),
                    'LeadStageId': int(row['LeadStageId']) if pd.notna(row['LeadStageId']) else None,
                    'CreatedOn': row['CreatedOn'].strftime('%Y-%m-%d %H:%M:%S'),
                    'ModifiedOn': row['ModifiedOn'].strftime('%Y-%m-%d %H:%M:%S')
                }
                
                if 'CountryName' in row and pd.notna(row['CountryName']):
                    detail['CountryName'] = str(row['CountryName'])
                
                lead_details.append(detail)
            
            self.engine.dispose()
            return {
                'status': 'success',
                'filter_type': filter_type,
                'filter_value': filter_value,
                'total_count': len(lead_ids),
                'lead_ids': lead_ids,
                'lead_details': lead_details,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def _get_status_distribution_internal(self, L: pd.DataFrame) -> list:
        """Internal: Get status distribution for complete analytics"""
        status_counts = L['Status'].value_counts().reset_index()
        status_counts.columns = ['status', 'count']
        total = len(L)
        
        return [
            {
                'status': row['status'],
                'count': int(row['count']),
                'percentage': round(row['count'] / total * 100, 1) if total > 0 else 0
            }
            for idx, row in status_counts.iterrows()
        ]
    
    def _get_top_statuses_internal(self, L: pd.DataFrame) -> list:
        """Internal: Get top 3 statuses for complete analytics"""
        status_counts = L['Status'].value_counts().head(3).reset_index()
        status_counts.columns = ['status', 'count']
        
        return [
            {'status': row['status'], 'count': int(row['count'])}
            for idx, row in status_counts.iterrows()
        ]
