export interface SuccessResponse {
  success: boolean;
  error_message?: string | null;
}

export interface PaginatedMeta {
  page: number;
  per_page: number;
  total_items: number;
  total_pages: number;
}

export interface DynamicWorkoutMetric {
  key: string;
  value: number;
  source: string;
  unit: string;
  display_name: string;
}

export interface DistributionBucket {
  min_value: number;
  max_value: number;
  time_in_seconds: number;
}

export interface DynamicMetricDistribution {
  key: string;
  display_name: string;
  unit: string;
  buckets: DistributionBucket[];
}

export interface ZoneData {
  zone_index: number;
  min_value: number;
  max_value: number;
  time_in_seconds: number;
}

export interface DynamicActivityZone {
  key: string;
  display_name: string;
  unit: string;
  reference_value?: number | null;
  reference_name?: string | null;
  zones: ZoneData[];
}

export interface PrivacySettings {
  profile_visibility: string;
  activity_visibility: string;
  biometrics_visibility: string;
}

export interface SummaryAthlete {
  id: string;
  first_name: string;
  last_name: string;
  profile_picture_url: string;
  city?: string;
  state?: string;
  country?: string;
}

export interface MeAthlete extends SummaryAthlete {
  username: string;
  email: string;
  city: string;
  state: string;
  country: string;
  gender?: string | null;
  birthdate?: string | null;
  weight_kg?: number | null;
  height_cm?: number | null;
  created: string;
  wearable_connected: boolean;
  privacy_settings: PrivacySettings;
}

export interface AthleteStats {
  current_ftp: number;
  threshold_pace: number;
  ytd_run_totals: number;
  all_time_run_totals: number;
}

export interface PersonalRecord {
  distance_name: string;
  time_in_seconds: number;
  activity_id: string;
  achieved_date: string;
}

export interface DetailedAthlete {
  id: string;
  username: string;
  first_name: string;
  last_name: string;
  city: string;
  state: string;
  country: string;
  profile_picture_url: string;
  created: string;
  wearable_connected: boolean;
  stats: AthleteStats;
  privacy_settings: PrivacySettings;
  personal_records: PersonalRecord[];
}

export interface SummaryActivity {
  id: string;
  athlete_id: string;
  name: string;
  activity_type: string;
  distance: number;
  moving_time: number;
  start_date: string;
  polyline_summary: string;
  device_name?: string;
  visibility: string;
  biometrics_visibility: string;
  like_count: number;
  comment_count: number;
  is_liked: boolean;
  metrics: DynamicWorkoutMetric[];
}

export interface DetailedActivity extends SummaryActivity {
  description: string;
  elapsed_time: number;
  polyline: string;
  device_name: string;
  gear_id?: string | null;
  perceived_exertion?: number | null;
  distributions: DynamicMetricDistribution[];
  zones: DynamicActivityZone[];
  laps: Lap[];
}

export interface Lap {
  id: string;
  lap_index: number;
  name: string;
  start_date: string;
  elapsed_time: number;
  moving_time: number;
  distance: number;
  average_speed: number;
}

export interface Split {
  index: number;
  distance: number;
  elapsed_time: number;
  elevation_difference: number;
  average_speed: number;
}

export interface Stream {
  metric_key: string;
  stream_type: string;
  data: number[];
  axis: number[];
  axis_type: string;
  original_size?: number;
  resolution?: string;
}

export interface PowerCurveValue {
  time_interval_seconds: number;
  power_value_watts: number;
}

export interface PowerCurve {
  curve_values: PowerCurveValue[];
}

export interface Comment {
  id: string;
  author: SummaryAthlete;
  text: string;
  created_at: string;
  like_count: number;
  is_liked: boolean;
}

export interface AthletePost {
  id: string;
  athlete_id: string;
  text: string;
  media_urls: string[];
  created_at: string;
  like_count: number;
  comment_count: number;
  is_liked: boolean;
}

export interface ClubPost {
  id: string;
  club_id: string;
  club: SummaryClub;
  author: SummaryAthlete;
  title: string;
  body: string;
  created_at: string;
  like_count: number;
  comment_count: number;
  is_liked: boolean;
}

export interface FeedItem {
  id: string;
  type: string;
  athlete: SummaryAthlete;
  created_at: string;
  activity_data?: SummaryActivity | null;
  post_data?: AthletePost | null;
  club_post_data?: ClubPost | null;
}

export interface Gear {
  id: string;
  name: string;
  brand_name: string;
  model_name: string;
  gear_type: string;
  is_primary: boolean;
  max_mileage: number;
  is_retired: boolean;
  initial_date?: string | null;
  created_date: string;
}

export interface SummarySegment {
  id: string;
  name: string;
  activity_type: string;
  distance: number;
  average_grade: number;
  start_latlng: number[];
  end_latlng: number[];
  is_starred: boolean;
}

export interface DetailedSegment extends SummarySegment {
  polyline: string;
  elevation_high: number;
  elevation_low: number;
  total_effort_count: number;
  total_athlete_count: number;
  star_count: number;
}

export interface SegmentEffort {
  id: string;
  segment_id: string;
  activity_id: string;
  elapsed_time: number;
  moving_time: number;
  start_date: string;
  average_heartrate?: number | null;
  average_power?: number | null;
  rank?: number | null;
}

export interface LeaderboardEntry {
  athlete_id: string;
  athlete_name: string;
  athlete_profile_picture_url: string;
  rank: number;
  elapsed_time: number;
  average_hr?: number | null;
  average_power?: number | null;
  achieved_date: string;
}

export interface SummaryRoute {
  id: string;
  name: string;
  distance: number;
  elevation_gain: number;
  polyline_summary: string;
  is_private: boolean;
  created_at: string;
}

export interface DetailedRoute extends SummaryRoute {
  athlete_id: string;
  description: string;
  polyline: string;
  waypoints: { lat: number; lng: number; elevation: number; name?: string | null }[];
  estimated_duration?: number | null;
}

export interface SummaryClub {
  id: string;
  name: string;
  profile_picture_url: string;
  member_count: number;
  is_private: boolean;
}

export interface DetailedClub extends SummaryClub {
  description: string;
  cover_photo_url: string;
  creator_id: string;
  created_at: string;
  admins: string[];
  tags: string[];
  is_member: boolean;
  viewer_role: string | null;
  has_pending_join_request: boolean;
  has_pending_invite: boolean;
}

export interface ClubLeaderboardEntry {
  rank: number;
  athlete_id: string;
  athlete: SummaryAthlete;
  distance: number;
  activity_count: number;
  longest_activity_id: string | null;
  longest_distance: number;
  avg_pace: number | null;
  elevation_gain: number;
}

export interface ClubLeaderboardSummary {
  rank: number | null;
  distance: number;
  activity_count: number;
  longest_distance: number;
  avg_pace: number | null;
  elevation_gain: number;
}

export interface NotificationTarget {
  kind: string;
  id: string;
  title: string;
  subtitle?: string | null;
  detail?: string | null;
  image_url?: string | null;
  activity_type?: string | null;
}

export interface Notification {
  id: string;
  type: string;
  is_read: boolean;
  created_at: string;
  sender_id?: string | null;
  sender?: SummaryAthlete | null;
  message: string;
  link_path: string;
  excerpt?: string | null;
  target?: NotificationTarget | null;
  payload: Record<string, string | undefined>;
}

export interface IntegrationStatus {
  provider: string;
  is_connected: boolean;
  connected_at?: string | null;
}

export interface ActiveSession {
  session_id: string;
  client_metadata: { user_agent: string; browser_name: string; browser_version: string; os_name: string };
  ip_address: string;
  location?: string | null;
  last_active_at: string;
  created_at: string;
  is_current: boolean;
}

export interface ConnectionSearchResult {
  athlete: SummaryAthlete;
  relationship_status: string;
  connection_degree: number;
  common_clubs_count: number;
  mutual_followers_count: number;
}

export interface EmailNotificationsSettings {
  comments: boolean;
  likes: boolean;
  follow_requests: boolean;
  club_invites: boolean;
}

export interface AthletePreferences {
  measurement_system: string;
  privacy_settings: PrivacySettings;
  theme: string;
  email_notifications: EmailNotificationsSettings;
}

export interface CursorPaginatedFeedResponse {
  next_cursor: string | null;
  items: FeedItem[];
}

export interface PaginatedActivitiesResponse {
  pagination: PaginatedMeta;
  items: SummaryActivity[];
}

export interface PaginatedCommentsResponse {
  pagination: PaginatedMeta;
  items: Comment[];
}

export interface PaginatedAthletesResponse {
  pagination: PaginatedMeta;
  items: SummaryAthlete[];
}

export interface PaginatedConnectionsResponse {
  pagination: PaginatedMeta;
  items: ConnectionSearchResult[];
}

export interface PaginatedSegmentsResponse {
  pagination: PaginatedMeta;
  items: SummarySegment[];
}

export interface PaginatedSegmentEffortsResponse {
  pagination: PaginatedMeta;
  items: SegmentEffort[];
}

export interface PaginatedLeaderboardResponse {
  pagination: PaginatedMeta;
  items: LeaderboardEntry[];
}

export interface PaginatedRoutesResponse {
  pagination: PaginatedMeta;
  items: SummaryRoute[];
}

export interface PaginatedClubsResponse {
  pagination: PaginatedMeta;
  items: SummaryClub[];
}

export interface PaginatedPostsResponse {
  pagination: PaginatedMeta;
  items: ClubPost[];
}

export interface PaginatedClubLeaderboardResponse {
  pagination: PaginatedMeta;
  items: ClubLeaderboardEntry[];
  viewer_summary: ClubLeaderboardSummary | null;
}

export interface PaginatedNotificationsResponse {
  pagination: PaginatedMeta;
  items: Notification[];
}
