import React from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import {
  Box,
  Drawer,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Typography,
  Chip,
  Divider,
  Avatar,
  useMediaQuery,
  useTheme,
} from '@mui/material';
import DashboardOutlined from '@mui/icons-material/DashboardOutlined';
import CurrencyBitcoinOutlined from '@mui/icons-material/CurrencyBitcoinOutlined';
import ScienceOutlined from '@mui/icons-material/ScienceOutlined';
import TimelineOutlined from '@mui/icons-material/TimelineOutlined';
import AccountBalanceWalletOutlined from '@mui/icons-material/AccountBalanceWalletOutlined';
import PlayArrowOutlined from '@mui/icons-material/PlayArrowOutlined';
import SettingsOutlined from '@mui/icons-material/SettingsOutlined';
import LogoutOutlined from '@mui/icons-material/LogoutOutlined';
import { useAuth } from '../../context/AuthContext';

export const SIDEBAR_WIDTH = 280;

const mainNavItems = [
  { label: 'Overview', path: '/', icon: <DashboardOutlined /> },
  { label: 'Portfolio', path: '/portfolio', icon: <AccountBalanceWalletOutlined /> },
  { label: 'Coins', path: '/coins', icon: <CurrencyBitcoinOutlined /> },
  { label: 'Backtest', path: '/backtest', icon: <ScienceOutlined /> },
  { label: 'Signals', path: '/signals', icon: <TimelineOutlined /> },
  { label: 'Paper Trading', path: '/paper-trading', icon: <PlayArrowOutlined /> },
];

interface SidebarProps {
  mobileOpen: boolean;
  onMobileClose: () => void;
}

const Sidebar: React.FC<SidebarProps> = ({ mobileOpen, onMobileClose }) => {
  const location = useLocation();
  const navigate = useNavigate();
  const { signOut, profile } = useAuth();
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));

  const isActive = (path: string) => {
    if (path === '/') return location.pathname === '/';
    return location.pathname.startsWith(path);
  };

  const handleNav = (path: string) => {
    navigate(path);
    if (isMobile) onMobileClose();
  };

  const handleLogout = async () => {
    await signOut();
  };

  const drawerContent = (
    <Box sx={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      {/* Logo Area */}
      <Box sx={{ p: 3, display: 'flex', alignItems: 'center', gap: 1 }}>
        <Typography variant="h6" sx={{ fontWeight: 700, color: 'text.primary' }}>
          TradeWise
        </Typography>
        <Chip label="v7.0" size="small" color="primary" variant="outlined" />
      </Box>

      <Divider />

      {/* Main Navigation */}
      <List sx={{ flex: 1, px: 2, py: 1 }}>
        {mainNavItems.map((item) => (
          <ListItem key={item.path} disablePadding sx={{ mb: 0.5 }}>
            <ListItemButton
              onClick={() => handleNav(item.path)}
              sx={{
                borderRadius: 1.5,
                py: 1,
                px: 2,
                color: isActive(item.path) ? 'primary.main' : 'text.secondary',
                backgroundColor: isActive(item.path) ? 'rgba(96, 165, 250, 0.08)' : 'transparent',
                '&:hover': {
                  backgroundColor: isActive(item.path)
                    ? 'rgba(96, 165, 250, 0.12)'
                    : 'rgba(255, 255, 255, 0.04)',
                },
              }}
            >
              <ListItemIcon
                sx={{
                  minWidth: 40,
                  color: isActive(item.path) ? 'primary.main' : 'text.secondary',
                }}
              >
                {item.icon}
              </ListItemIcon>
              <ListItemText
                primary={item.label}
                primaryTypographyProps={{
                  fontSize: '0.875rem',
                  fontWeight: isActive(item.path) ? 600 : 400,
                }}
              />
            </ListItemButton>
          </ListItem>
        ))}
      </List>

      <Divider />

      {/* Profile Section */}
      <Box
        sx={{
          px: 2,
          py: 2,
          display: 'flex',
          alignItems: 'center',
          gap: 1.5,
          cursor: 'pointer',
          '&:hover': { backgroundColor: 'rgba(255, 255, 255, 0.04)' },
        }}
        onClick={() => handleNav('/settings')}
      >
        <Avatar
          src={profile?.avatar_url || undefined}
          sx={{ width: 36, height: 36, bgcolor: 'primary.main', fontSize: '0.875rem' }}
        >
          {profile?.display_name?.charAt(0)?.toUpperCase() || profile?.username?.charAt(0)?.toUpperCase() || 'U'}
        </Avatar>
        <Box sx={{ flex: 1, minWidth: 0 }}>
          <Typography
            variant="body2"
            sx={{ fontWeight: 600, color: 'text.primary', lineHeight: 1.3, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}
          >
            {profile?.display_name || profile?.username || 'User'}
          </Typography>
          <Typography
            variant="caption"
            sx={{ color: 'text.secondary', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', display: 'block' }}
          >
            {profile?.email || ''}
          </Typography>
        </Box>
      </Box>

      <Divider />

      {/* Bottom: Settings + Logout */}
      <List sx={{ px: 2, py: 1 }}>
        <ListItem disablePadding sx={{ mb: 0.5 }}>
          <ListItemButton
            onClick={() => handleNav('/settings')}
            sx={{
              borderRadius: 1.5,
              py: 1,
              px: 2,
              color: isActive('/settings') ? 'primary.main' : 'text.secondary',
              backgroundColor: isActive('/settings') ? 'rgba(96, 165, 250, 0.08)' : 'transparent',
              '&:hover': { backgroundColor: 'rgba(255, 255, 255, 0.04)' },
            }}
          >
            <ListItemIcon
              sx={{
                minWidth: 40,
                color: isActive('/settings') ? 'primary.main' : 'text.secondary',
              }}
            >
              <SettingsOutlined />
            </ListItemIcon>
            <ListItemText
              primary="Settings"
              primaryTypographyProps={{ fontSize: '0.875rem' }}
            />
          </ListItemButton>
        </ListItem>
        <ListItem disablePadding>
          <ListItemButton
            onClick={handleLogout}
            sx={{
              borderRadius: 1.5,
              py: 1,
              px: 2,
              color: 'text.secondary',
              '&:hover': { backgroundColor: 'rgba(239, 68, 68, 0.08)', color: 'error.main' },
            }}
          >
            <ListItemIcon sx={{ minWidth: 40, color: 'inherit' }}>
              <LogoutOutlined />
            </ListItemIcon>
            <ListItemText
              primary="Logout"
              primaryTypographyProps={{ fontSize: '0.875rem' }}
            />
          </ListItemButton>
        </ListItem>
      </List>
    </Box>
  );

  return (
    <Box component="nav" sx={{ width: { md: SIDEBAR_WIDTH }, flexShrink: { md: 0 } }}>
      {/* Mobile drawer */}
      <Drawer
        variant="temporary"
        open={mobileOpen}
        onClose={onMobileClose}
        ModalProps={{ keepMounted: true }}
        sx={{
          display: { xs: 'block', md: 'none' },
          '& .MuiDrawer-paper': { width: SIDEBAR_WIDTH },
        }}
      >
        {drawerContent}
      </Drawer>

      {/* Desktop drawer */}
      <Drawer
        variant="permanent"
        sx={{
          display: { xs: 'none', md: 'block' },
          '& .MuiDrawer-paper': { width: SIDEBAR_WIDTH },
        }}
        open
      >
        {drawerContent}
      </Drawer>
    </Box>
  );
};

export default Sidebar;
