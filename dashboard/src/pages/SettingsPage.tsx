import React, { useState } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  TextField,
  Button,
  Avatar,
  Divider,
  Alert,
} from '@mui/material';
import SaveOutlined from '@mui/icons-material/SaveOutlined';
import { useAuth } from '../context/AuthContext';

const SettingsPage: React.FC = () => {
  const { profile, updateCapital } = useAuth();

  const [capital, setCapital] = useState<number>(profile?.capital || 10000);
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);

  const handleSaveCapital = async () => {
    if (capital <= 0) return;
    setSaving(true);
    setMessage(null);
    const success = await updateCapital(capital);
    setSaving(false);
    setMessage(
      success
        ? { type: 'success', text: 'Capital updated successfully.' }
        : { type: 'error', text: 'Failed to update capital.' }
    );
  };

  return (
    <Box>
      <Typography variant="h5" sx={{ fontWeight: 700, mb: 3, color: 'text.primary' }}>
        Settings
      </Typography>

      <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: '1fr 1fr' }, gap: 3 }}>
        {/* Profile Card */}
        <Card>
          <CardContent sx={{ p: 3 }}>
            <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 2 }}>
              Profile
            </Typography>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 3 }}>
              <Avatar
                src={profile?.avatar_url || undefined}
                sx={{ width: 56, height: 56, bgcolor: 'primary.main', fontSize: '1.25rem' }}
              >
                {profile?.display_name?.charAt(0)?.toUpperCase() || profile?.username?.charAt(0)?.toUpperCase() || 'U'}
              </Avatar>
              <Box>
                <Typography variant="body1" sx={{ fontWeight: 600, color: 'text.primary' }}>
                  {profile?.display_name || profile?.username || 'User'}
                </Typography>
                <Typography variant="body2" sx={{ color: 'text.secondary' }}>
                  {profile?.email}
                </Typography>
              </Box>
            </Box>

            <Divider sx={{ my: 2 }} />

            <Box sx={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 2 }}>
              <Box>
                <Typography variant="caption" sx={{ color: 'text.secondary' }}>Username</Typography>
                <Typography variant="body2" sx={{ color: 'text.primary', fontWeight: 500 }}>
                  {profile?.username || '-'}
                </Typography>
              </Box>
              <Box>
                <Typography variant="caption" sx={{ color: 'text.secondary' }}>Experience</Typography>
                <Typography variant="body2" sx={{ color: 'text.primary', fontWeight: 500 }}>
                  {profile?.experience_level || '-'}
                </Typography>
              </Box>
              <Box>
                <Typography variant="caption" sx={{ color: 'text.secondary' }}>Default Trade Type</Typography>
                <Typography variant="body2" sx={{ color: 'text.primary', fontWeight: 500 }}>
                  {profile?.default_trade_type || '-'}
                </Typography>
              </Box>
              <Box>
                <Typography variant="caption" sx={{ color: 'text.secondary' }}>Member Since</Typography>
                <Typography variant="body2" sx={{ color: 'text.primary', fontWeight: 500 }}>
                  {profile?.created_at ? new Date(profile.created_at).toLocaleDateString() : '-'}
                </Typography>
              </Box>
            </Box>
          </CardContent>
        </Card>

        {/* Capital Management Card */}
        <Card>
          <CardContent sx={{ p: 3 }}>
            <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 2 }}>
              Capital Management
            </Typography>
            <Typography variant="body2" sx={{ color: 'text.secondary', mb: 3 }}>
              Set your trading capital for portfolio tracking and position sizing.
            </Typography>

            <TextField
              label="Trading Capital (USD)"
              type="number"
              value={capital}
              onChange={(e) => setCapital(Number(e.target.value))}
              fullWidth
              size="small"
              inputProps={{ min: 100, step: 1000 }}
              sx={{ mb: 2 }}
            />

            <Button
              variant="contained"
              startIcon={<SaveOutlined />}
              onClick={handleSaveCapital}
              disabled={saving || capital <= 0}
              fullWidth
            >
              {saving ? 'Saving...' : 'Save Capital'}
            </Button>

            {message && (
              <Alert severity={message.type} sx={{ mt: 2 }}>
                {message.text}
              </Alert>
            )}
          </CardContent>
        </Card>
      </Box>
    </Box>
  );
};

export default SettingsPage;
