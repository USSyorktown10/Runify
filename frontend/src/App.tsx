import { BrowserRouter, Navigate, Route, Routes, useParams } from "react-router-dom";
import { ProtectedRoute } from "@/components/ProtectedRoute";
import { AppShell, AuthLayout, PublicLayout } from "@/layouts/AppShell";
import { LandingPage } from "@/pages/LandingPage";
import { FeedPage } from "@/pages/FeedPage";
import { NotFoundPage } from "@/pages/NotFoundPage";
import { NotificationsPage } from "@/pages/NotificationsPage";
import { SearchPage } from "@/pages/SearchPage";
import { LoginPage } from "@/pages/auth/LoginPage";
import { SignupPage } from "@/pages/auth/SignupPage";
import { ForgotPasswordPage } from "@/pages/auth/ForgotPasswordPage";
import { ResetPasswordPage } from "@/pages/auth/ResetPasswordPage";
import { VerifyEmailPage } from "@/pages/auth/VerifyEmailPage";
import { OAuthCallbackPage } from "@/pages/auth/OAuthCallbackPage";
import { ActivitiesPage } from "@/pages/activities/ActivitiesPage";
import { CreateActivityPage } from "@/pages/activities/CreateActivityPage";
import { UploadActivityPage } from "@/pages/activities/UploadActivityPage";
import { ActivityDetailPage } from "@/pages/activities/ActivityDetailPage";
import { EditActivityPage } from "@/pages/activities/EditActivityPage";
import { CropActivityPage } from "@/pages/activities/CropActivityPage";
import { AthleteProfilePage } from "@/pages/athletes/AthleteProfilePage";
import { AthleteFeedPage } from "@/pages/athletes/AthleteFeedPage";
import { AthleteFollowersPage, AthleteFollowingPage } from "@/pages/athletes/AthleteListPage";
import { AthleteSegmentsPage } from "@/pages/athletes/AthleteSegmentsPage";
import { SettingsPage } from "@/pages/settings/SettingsPage";
import { ProfileSettingsPage } from "@/pages/settings/ProfileSettingsPage";
import { PreferencesSettingsPage } from "@/pages/settings/PreferencesSettingsPage";
import { StatsSettingsPage } from "@/pages/settings/StatsSettingsPage";
import { GearSettingsPage } from "@/pages/settings/GearSettingsPage";
import { GearDetailPage } from "@/pages/settings/GearDetailPage";
import { IntegrationsSettingsPage } from "@/pages/settings/IntegrationsSettingsPage";
import { SessionsSettingsPage } from "@/pages/settings/SessionsSettingsPage";
import { PrivacySettingsPage } from "@/pages/settings/PrivacySettingsPage";
import { FollowRequestsPage } from "@/pages/settings/FollowRequestsPage";
import { SegmentsPage } from "@/pages/segments/SegmentsPage";
import { SegmentDetailPage } from "@/pages/segments/SegmentDetailPage";
import { SegmentLeaderboardPage } from "@/pages/segments/SegmentLeaderboardPage";
import { SegmentEffortsPage } from "@/pages/segments/SegmentEffortsPage";
import { RoutesPage } from "@/pages/routes/RoutesPage";
import { CreateRoutePage } from "@/pages/routes/CreateRoutePage";
import { RouteDetailPage } from "@/pages/routes/RouteDetailPage";
import { EditRoutePage } from "@/pages/routes/EditRoutePage";
import { ClubsPage } from "@/pages/clubs/ClubsPage";
import { CreateClubPage } from "@/pages/clubs/CreateClubPage";
import { ClubDetailPage } from "@/pages/clubs/ClubDetailPage";
import { ClubSettingsPage } from "@/pages/clubs/ClubSettingsPage";
import { ClubJoinRequestsPage } from "@/pages/clubs/ClubAdminPages";

function ClubTabRedirect({ tab }: { tab: string }) {
  const { id } = useParams<{ id: string }>();
  return <Navigate to={`/clubs/${id}?tab=${tab}`} replace />;
}

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<PublicLayout />}>
          <Route path="/" element={<LandingPage />} />
        </Route>

        <Route path="/login" element={<LoginPage />} />

        <Route element={<AuthLayout />}>
          <Route path="/signup" element={<SignupPage />} />
          <Route path="/forgot-password" element={<ForgotPasswordPage />} />
          <Route path="/reset-password" element={<ResetPasswordPage />} />
          <Route path="/verify-email" element={<VerifyEmailPage />} />
        </Route>

        <Route path="/auth/google/callback" element={<OAuthCallbackPage provider="google" />} />
        <Route path="/auth/apple/callback" element={<OAuthCallbackPage provider="apple" />} />

        <Route element={<ProtectedRoute />}>
          <Route element={<AppShell />}>
            <Route path="/feed" element={<FeedPage />} />
            <Route path="/activities" element={<ActivitiesPage />} />
            <Route path="/activities/new" element={<CreateActivityPage />} />
            <Route path="/activities/upload" element={<UploadActivityPage />} />
            <Route path="/activities/:id" element={<ActivityDetailPage />} />
            <Route path="/activities/:id/edit" element={<EditActivityPage />} />
            <Route path="/activities/:id/crop" element={<CropActivityPage />} />
            <Route path="/athletes/:id" element={<AthleteProfilePage />} />
            <Route path="/athletes/:id/feed" element={<AthleteFeedPage />} />
            <Route path="/athletes/:id/followers" element={<AthleteFollowersPage />} />
            <Route path="/athletes/:id/following" element={<AthleteFollowingPage />} />
            <Route path="/athletes/:id/segments" element={<AthleteSegmentsPage />} />
            <Route path="/search" element={<SearchPage />} />
            <Route path="/segments" element={<SegmentsPage />} />
            <Route path="/segments/:id" element={<SegmentDetailPage />} />
            <Route path="/segments/:id/leaderboard" element={<SegmentLeaderboardPage />} />
            <Route path="/segments/:id/efforts" element={<SegmentEffortsPage />} />
            <Route path="/routes" element={<RoutesPage />} />
            <Route path="/routes/new" element={<CreateRoutePage />} />
            <Route path="/routes/:id" element={<RouteDetailPage />} />
            <Route path="/routes/:id/edit" element={<EditRoutePage />} />
            <Route path="/clubs" element={<ClubsPage />} />
            <Route path="/clubs/new" element={<CreateClubPage />} />
            <Route path="/clubs/:id" element={<ClubDetailPage />} />
            <Route path="/clubs/:id/members" element={<ClubTabRedirect tab="members" />} />
            <Route path="/clubs/:id/settings" element={<ClubSettingsPage />} />
            <Route path="/clubs/:id/join-requests" element={<ClubJoinRequestsPage />} />
            <Route path="/clubs/:id/leaderboard" element={<ClubTabRedirect tab="leaderboard" />} />
            <Route path="/notifications" element={<NotificationsPage />} />
            <Route path="/settings" element={<SettingsPage />} />
            <Route path="/settings/profile" element={<ProfileSettingsPage />} />
            <Route path="/settings/preferences" element={<PreferencesSettingsPage />} />
            <Route path="/settings/stats" element={<StatsSettingsPage />} />
            <Route path="/settings/gear" element={<GearSettingsPage />} />
            <Route path="/settings/gear/:id" element={<GearDetailPage />} />
            <Route path="/settings/integrations" element={<IntegrationsSettingsPage />} />
            <Route path="/settings/sessions" element={<SessionsSettingsPage />} />
            <Route path="/settings/privacy" element={<PrivacySettingsPage />} />
            <Route path="/settings/follow-requests" element={<FollowRequestsPage />} />
          </Route>
        </Route>

        <Route path="/home" element={<Navigate to="/feed" replace />} />
        <Route path="*" element={<NotFoundPage />} />
      </Routes>
    </BrowserRouter>
  );
}
