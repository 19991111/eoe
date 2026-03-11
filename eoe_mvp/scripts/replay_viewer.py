"""
PyGame回放播放器 - 调试人工生命的可视化工具

功能：
- 播放/暂停/单步
- 显示Agent传感器视锥
- 显示状态HUD（能量、胃部、携带）
- 显示轨迹线
"""
import os
os.environ['SDL_AUDIODRIVER'] = 'dummy'  # 禁用音频，避免ALSA错误

import pygame
import json
import sys

# 配置
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 800
WORLD_SIZE = 100
SCALE = WINDOW_WIDTH / WORLD_SIZE
BG_COLOR = (20, 20, 30)
GRID_COLOR = (40, 40, 50)

# 颜色
SEASON_COLORS = {
    'summer': (255, 200, 100),
    'winter': (100, 150, 255)
}

class ReplayViewer:
    def __init__(self, replay_file):
        self.replay_file = replay_file
        self.load_data()
        
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("EOE 回放播放器 - 障碍物绕行实验")
        self.clock = pygame.time.Clock()
        
        # 播放控制
        self.current_frame = 0
        self.playing = False
        self.speed = 1  # 播放速度
        self.show_sensors = True
        self.show_trails = True
        self.follow_agent = None  # 跟踪的Agent ID
        
        # 轨迹缓存
        self.trails = {}  # agent_id -> [(x, y), ...]
        
    def load_data(self):
        """加载JSON回放数据"""
        with open(self.replay_file, 'r') as f:
            self.data = json.load(f)
        
        self.metadata = self.data.get('metadata', {})
        self.frames = self.data.get('frames', [])
        print(f"加载完成: {len(self.frames)} 帧")
        print(f"实验: {self.metadata.get('experiment', 'N/A')}")
    
    def world_to_screen(self, x, y):
        """世界坐标转屏幕坐标"""
        return int(x * SCALE), int(y * SCALE)
    
    def draw_grid(self):
        """绘制网格"""
        for i in range(0, WORLD_SIZE + 1, 10):
            x, _ = self.world_to_screen(i, 0)
            _, y = self.world_to_screen(0, i)
            pygame.draw.line(self.screen, GRID_COLOR, (x, 0), (x, WINDOW_HEIGHT), 1)
            pygame.draw.line(self.screen, GRID_COLOR, (0, y), (WINDOW_WIDTH, y), 1)
    
    def draw_obstacles(self, obstacles):
        """绘制障碍物（柱子）"""
        for obs in obstacles:
            cx, cy = self.world_to_screen(obs['x'], obs['y'])
            r = int(obs['r'] * SCALE)
            pygame.draw.circle(self.screen, (80, 80, 100), (cx, cy), r)
            pygame.draw.circle(self.screen, (120, 120, 140), (cx, cy), r, 2)
    
    def draw_food(self, food_list):
        """绘制食物"""
        for food in food_list:
            fx, fy = self.world_to_screen(food['x'], food['y'])
            pygame.draw.circle(self.screen, (50, 200, 100), (fx, fy), 4)
            # 光晕效果
            pygame.draw.circle(self.screen, (50, 200, 100), (fx, fy), 7, 1)
    
    def draw_nest(self):
        """绘制巢穴"""
        nx, ny = self.world_to_screen(50, 50)
        pygame.draw.circle(self.screen, (100, 100, 200), (nx, ny), 10)
        pygame.draw.circle(self.screen, (150, 150, 255), (nx, ny), 15, 2)
    
    def draw_sensors(self, agent):
        """绘制传感器视锥"""
        x, y = self.world_to_screen(agent['x'], agent['y'])
        
        # 简化的扇形视锥
        sensor_length = 25 * SCALE
        angle_range = 60  # 度
        
        # 绘制扇形
        points = [(x, y)]
        for angle in range(-angle_range, angle_range + 1, 10):
            rad = angle * 3.14159 / 180
            end_x = x + sensor_length * 0.5 * (1 + 0.5 * abs(angle) / angle_range)
            end_y = y - sensor_length * 0.5
            points.append((end_x, end_y))
        
        # 半透明填充
        surf = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
        pygame.draw.polygon(surf, (255, 255, 0, 30), points)
        self.screen.blit(surf, (0, 0))
        
        # 轮廓
        pygame.draw.polygon(self.screen, (255, 255, 0, 100), points, 1)
    
    def draw_agent(self, agent, trail=None):
        """绘制单个Agent"""
        x, y = self.world_to_screen(agent['x'], agent['y'])
        
        # 轨迹线
        if trail and len(trail) > 1:
            if len(trail) > 50:
                trail = trail[-50:]  # 只显示最近50步
            for i in range(1, len(trail)):
                tx, ty = self.world_to_screen(trail[i-1][0], trail[i-1][1])
                cx, cy = self.world_to_screen(trail[i][0], trail[i][1])
                alpha = int(255 * i / len(trail))
                pygame.draw.line(self.screen, (100, 200, 255, alpha), (tx, ty), (cx, cy), 2)
        
        # 主体颜色根据状态变化
        if not agent['is_alive']:
            color = (80, 80, 80)  # 死亡灰色
        elif agent['food_carried'] > 0:
            color = (255, 200, 0)  # 携带食物 - 金色
        elif agent['food_in_stomach'] > 0:
            color = (0, 200, 100)  # 吃饱 - 绿色
        else:
            color = (200, 50, 50)  # 饥饿 - 红色
        
        # 绘制身体
        pygame.draw.circle(self.screen, color, (x, y), 6)
        
        # 能量环
        energy_ratio = agent['energy'] / 200
        energy_color = (int(255 * (1-energy_ratio)), int(255 * energy_ratio), 0)
        pygame.draw.circle(self.screen, energy_color, (x, y), 9, 2)
        
        # 状态图标
        if agent['food_carried'] > 0:
            # 手里拿东西
            pygame.draw.circle(self.screen, (255, 255, 0), (x+5, y-5), 3)
        if agent['food_stored'] > 0:
            # 存贮 - 巢穴图标
            nx, ny = self.world_to_screen(50, 50)
            if (agent['x']-50)**2 + (agent['y']-50)**2 < 15**2:
                pygame.draw.circle(self.screen, (0, 255, 255), (x, y-10), 3)
    
    def draw_hud(self):
        """绘制HUD信息"""
        font = pygame.font.Font(None, 20)
        
        # 当前帧信息
        frame_info = f"Frame: {self.current_frame} / {len(self.frames)-1}"
        surf = font.render(frame_info, True, (200, 200, 200))
        self.screen.blit(surf, (10, 10))
        
        # 季节信息
        frame_data = self.frames[self.current_frame]
        season = frame_data.get('season', 'summer')
        season_surf = font.render(f"Season: {season}", True, SEASON_COLORS.get(season, (255, 255, 255)))
        self.screen.blit(season_surf, (10, 35))
        
        # Agent数量
        n_agents = len(frame_data['agents'])
        agents_surf = font.render(f"Agents: {n_agents}", True, (200, 200, 200))
        self.screen.blit(agents_surf, (10, 60))
        
        # 控制提示
        controls = "SPACE: 播放/暂停 | LEFT/RIGHT: 单步 | S: 传感器 | T: 轨迹"
        ctrl_surf = font.render(controls, True, (150, 150, 150))
        self.screen.blit(ctrl_surf, (10, WINDOW_HEIGHT - 25))
        
        # 当前最高分Agent信息
        if frame_data['agents']:
            best = max(frame_data['agents'], key=lambda a: a.get('energy', 0))
            info = f"Best Agent - Energy: {best['energy']:.0f}, Carry: {best['food_carried']}, Store: {best['food_stored']}"
            info_surf = font.render(info, True, (255, 200, 100))
            self.screen.blit(info_surf, (10, 85))
    
    def update_trails(self, agents):
        """更新轨迹"""
        for agent in agents:
            aid = agent['id']
            if agent['is_alive']:
                if aid not in self.trails:
                    self.trails[aid] = []
                self.trails[aid].append((agent['x'], agent['y']))
            else:
                # 死亡的Agent轨迹保留但不增加
                if aid in self.trails:
                    self.trails[aid].append(None)  # 标记死亡
    
    def render(self):
        """渲染当前帧"""
        self.screen.fill(BG_COLOR)
        self.draw_grid()
        
        frame_data = self.frames[self.current_frame]
        
        # 绘制障碍物
        self.draw_obstacles(frame_data.get('obstacles', [{'x': 50, 'y': 55, 'r': 8}]))
        
        # 绘制食物
        self.draw_food(frame_data.get('food', []))
        
        # 绘制巢穴
        self.draw_nest()
        
        # 更新和绘制Agent
        agents = frame_data['agents']
        self.update_trails(agents)
        
        for agent in agents:
            trail = self.trails.get(agent['id'], []) if self.show_trails else None
            self.draw_agent(agent, trail)
            
            # 传感器视锥
            if self.show_sensors and agent['is_alive']:
                self.draw_sensors(agent)
        
        # HUD
        self.draw_hud()
        
        pygame.display.flip()
    
    def run(self):
        """主循环"""
        running = True
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.playing = not self.playing
                    elif event.key == pygame.K_LEFT:
                        self.current_frame = max(0, self.current_frame - 1)
                    elif event.key == pygame.K_RIGHT:
                        self.current_frame = min(len(self.frames) - 1, self.current_frame + 1)
                    elif event.key == pygame.K_s:
                        self.show_sensors = not self.show_sensors
                    elif event.key == pygame.K_t:
                        self.show_trails = not self.show_trails
                    elif event.key == pygame.K_ESCAPE:
                        running = False
            
            if self.playing:
                self.current_frame = (self.current_frame + self.speed) % len(self.frames)
                self.clock.tick(30)  # 30 FPS
            else:
                self.clock.tick(60)
            
            self.render()
        
        pygame.quit()


def main():
    if len(sys.argv) < 2:
        print("用法: python replay_viewer.py <replay_file.json>")
        print("示例: python replay_viewer.py replay_data.json")
        sys.exit(1)
    
    replay_file = sys.argv[1]
    if not os.path.exists(replay_file):
        print(f"文件不存在: {replay_file}")
        sys.exit(1)
    
    viewer = ReplayViewer(replay_file)
    viewer.run()


if __name__ == '__main__':
    main()
