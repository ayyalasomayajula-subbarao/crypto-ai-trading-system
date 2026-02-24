import React from 'react';
import {
  AppBar,
  Toolbar,
  IconButton,
  Box,
  Chip,
  useMediaQuery,
  useTheme,
} from '@mui/material';
import MenuIcon from '@mui/icons-material/Menu';
import { SIDEBAR_WIDTH } from './Sidebar';

interface TopBarProps {
  onMenuToggle: () => void;
}

const TopBar: React.FC<TopBarProps> = ({ onMenuToggle }) => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));

  return (
    <AppBar
      position="fixed"
      sx={{
        width: { md: `calc(100% - ${SIDEBAR_WIDTH}px)` },
        ml: { md: `${SIDEBAR_WIDTH}px` },
      }}
    >
      <Toolbar sx={{ justifyContent: 'space-between' }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          {isMobile && (
            <IconButton color="inherit" onClick={onMenuToggle} edge="start">
              <MenuIcon />
            </IconButton>
          )}
        </Box>

        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          <Chip
            size="small"
            label="Live"
            color="success"
            variant="outlined"
            sx={{ fontSize: '0.75rem' }}
          />
        </Box>
      </Toolbar>
    </AppBar>
  );
};

export default TopBar;
