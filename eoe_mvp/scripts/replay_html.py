"""
生成HTML可视化 - 可以在浏览器查看
"""
import json
import sys

def generate_html(replay_file, output_file="replay_visualization.html"):
    with open(replay_file, 'r') as f:
        data = json.load(f)
    
    frames = data.get('frames', [])
    metadata = data.get('metadata', {})
    
    # 采样关键帧（减少文件大小）
    sample_rate = 5
    sampled_frames = frames[::sample_rate]
    
    html = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>EOE 演化可视化 - {metadata.get('experiment', 'Demo')}</title>
    <style>
        body {{ 
            background: #0a0a12; 
            color: #ccc; 
            font-family: 'Courier New', monospace;
            margin: 0; padding: 20px;
            display: flex; flex-direction: column; align-items: center;
        }}
        h1 {{ color: #ffd700; margin-bottom: 10px; }}
        #canvas {{
            border: 2px solid #334;
            background: #14141e;
            image-rendering: pixelated;
        }}
        #controls {{
            margin: 15px 0;
            display: flex; gap: 10px; align-items: center;
        }}
        button {{
            background: #223; color: #fff; border: 1px solid #445;
            padding: 8px 16px; cursor: pointer; font-family: inherit;
        }}
        button:hover {{ background: #334; }}
        #info {{
            color: #888; font-size: 14px;
        }}
        .legend {{
            display: flex; gap: 20px; margin: 10px 0;
        }}
        .legend-item {{ display: flex; align-items: center; gap: 5px; }}
        .dot {{ width: 12px; height: 12px; border-radius: 50%; }}
    </style>
</head>
<body>
    <h1>🧬 EOE 障碍物绕行实验</h1>
    <div class="legend">
        <div class="legend-item"><div class="dot" style="background:#c83232"></div>饥饿</div>
        <div class="legend-item"><div class="dot" style="background:#32c864"></div>吃饱</div>
        <div class="legend-item"><div class="dot" style="background:#ffd700"></div>携带</div>
        <div class="legend-item"><div class="dot" style="background:#444"></div>死亡</div>
        <div class="legend-item"><div class="dot" style="background:#505068"></div>障碍物</div>
    </div>
    <canvas id="canvas" width="600" height="600"></canvas>
    <div id="controls">
        <button onclick="play()">▶ 播放</button>
        <button onclick="pause()">⏸ 暂停</button>
        <button onclick="step(-1)">◀ 后退</button>
        <button onclick="step(1)">前进 ▶</button>
        <input type="range" id="slider" min="0" max="{len(sampled_frames)-1}" value="0" oninput="goto(parseInt(this.value))">
        <span id="frame-info">0 / {len(sampled_frames)-1}</span>
    </div>
    <div id="info">Gen: <span id="gen">0</span> | Season: <span id="season">summer</span> | Agents: <span id="nagents">0</span></div>
    
    <script>
    const frames = {json.dumps(sampled_frames)};
    const SCALE = 6;
    const WORLD = 100;
    const CX = 300, CY = 300;
    
    let current = 0;
    let playing = false;
    let interval = null;
    
    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');
    
    function worldToCanvas(x, y) {{
        return [CX + (x - WORLD/2) * SCALE, CY + (y - WORLD/2) * SCALE];
    }}
    
    function draw() {{
        const frame = frames[current];
        
        // 清空
        ctx.fillStyle = '#14141e';
        ctx.fillRect(0, 0, 600, 600);
        
        // 网格
        ctx.strokeStyle = '#1e1e2a';
        for(let i=0; i<=100; i+=10) {{
            let [x, y] = worldToCanvas(i, 0);
            let [, y2] = worldToCanvas(i, 100);
            ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x, y2); ctx.stroke();
            [x, y] = worldToCanvas(0, i);
            [, y2] = worldToCanvas(100, i);
            ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x, y2); ctx.stroke();
        }}
        
        // 障碍物
        for(let obs of frame.obstacles || []) {{
            let [x, y] = worldToCanvas(obs.x, obs.y);
            ctx.fillStyle = '#505068';
            ctx.beginPath(); ctx.arc(x, y, obs.r * SCALE, 0, 6.28); ctx.fill();
            ctx.strokeStyle = '#707088';
            ctx.lineWidth = 2; ctx.stroke();
        }}
        
        // 巢穴
        let [nx, ny] = worldToCanvas(50, 50);
        ctx.fillStyle = '#3264a8';
        ctx.beginPath(); ctx.arc(nx, ny, 10, 0, 6.28); ctx.fill();
        ctx.strokeStyle = '#4890d0'; ctx.lineWidth = 2; ctx.stroke();
        
        // 食物
        for(let f of frame.food || []) {{
            let [fx, fy] = worldToCanvas(f.x, f.y);
            ctx.fillStyle = '#32c864';
            ctx.beginPath(); ctx.arc(fx, fy, 4, 0, 6.28); ctx.fill();
        }}
        
        // Agent
        for(let a of frame.agents || []) {{
            let [ax, ay] = worldToCanvas(a.x, a.y);
            
            let color;
            if(!a.is_alive) color = '#444';
            else if(a.food_carried > 0) color = '#ffd700';
            else if(a.food_in_stomach > 0) color = '#32c864';
            else color = '#c83232';
            
            ctx.fillStyle = color;
            ctx.beginPath(); ctx.arc(ax, ay, 5, 0, 6.28); ctx.fill();
            
            // 能量环
            let energy = Math.min(1, a.energy / 200);
            ctx.strokeStyle = `rgb(${{255*(1-energy)}}, ${{255*energy}}, 0)`;
            ctx.lineWidth = 2;
            ctx.beginPath(); ctx.arc(ax, ay, 8, 0, 6.28); ctx.stroke();
            
            // 携带/存储图标
            if(a.food_carried > 0) {{
                ctx.fillStyle = '#ffd700';
                ctx.beginPath(); ctx.arc(ax+4, ay-4, 3, 0, 6.28); ctx.fill();
            }}
        }}
        
        // 更新UI
        document.getElementById('frame-info').textContent = current + ' / ' + (frames.length-1);
        document.getElementById('gen').textContent = frame.gen || 0;
        document.getElementById('season').textContent = frame.season || 'summer';
        document.getElementById('nagents').textContent = (frame.agents || []).length;
        document.getElementById('slider').value = current;
    }}
    
    function play() {{
        if(playing) return;
        playing = true;
        interval = setInterval(() => {{
            current = (current + 1) % frames.length;
            draw();
        }}, 50);
    }}
    
    function pause() {{
        playing = false;
        if(interval) clearInterval(interval);
    }}
    
    function step(dir) {{
        pause();
        current = Math.max(0, Math.min(frames.length-1, current + dir));
        draw();
    }}
    
    function goto(n) {{
        pause();
        current = n;
        draw();
    }}
    
    // 键盘控制
    document.addEventListener('keydown', e => {{
        if(e.key === ' ') play();
        if(e.key === 'ArrowLeft') step(-1);
        if(e.key === 'ArrowRight') step(1);
    }});
    
    draw();
    </script>
</body>
</html>'''
    
    with open(output_file, 'w') as f:
        f.write(html)
    
    print(f"✅ HTML可视化已生成: {output_file}")
    print(f"   帧数: {len(sampled_frames)} (采样自{len(frames)}帧)")
    return output_file

if __name__ == '__main__':
    import sys
    f = sys.argv[1] if len(sys.argv) > 1 else 'replay_v093_demo.json'
    generate_html(f)
