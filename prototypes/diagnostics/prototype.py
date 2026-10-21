"""Standalone UI study. Synthetic data only; imports no SOMA business modules.
Run: python prototype.py       Capture: python prototype.py --capture
"""
import os
import sys
import math
from pathlib import Path
if '--capture' in sys.argv:
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'
import numpy as np
from PySide6.QtCore import Qt, QPointF, QRectF
from PySide6.QtGui import QColor, QFont, QFontDatabase, QImage, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (QApplication, QWidget, QLabel, QPushButton, QComboBox,
                             QCheckBox, QVBoxLayout, QHBoxLayout, QFrame, QButtonGroup)

CYAN, ORANGE = '#39D5E8', '#FFAB55'
angles = np.arange(360, dtype=float)
delta = .07*np.sin(np.deg2rad(angles*2)) + .11*np.exp(-((angles-135)/22)**2) + .015
valid = ~((angles >= 250) & (angles <= 270))
values = delta[valid]

def label(text, size=10, color='#536071', bold=False):
    w = QLabel(text); w.setWordWrap(True)
    w.setStyleSheet(f'color:{color};font-size:{size}pt;font-weight:{600 if bold else 400};')
    return w

class Plot(QWidget):
    def __init__(self, owner, kind='image'):
        super().__init__(); self.owner=owner; self.kind=kind; self.zoom=1.; self.pan=QPointF(); self.last=None
        self.setMouseTracking(True); self.setMinimumHeight(150)
        y,x=np.mgrid[:600,:800]; r=np.hypot(x-400,y-300)
        rng=np.random.default_rng(48)
        gray=np.clip(43+112/(1+np.exp(np.clip(-(r-195)/2,-60,60)))+rng.normal(0,3,r.shape),0,255).astype('uint8')
        self.image=QImage(gray.data,800,600,800,QImage.Format_Grayscale8).copy()
    def text(self,p,x,y,s,color='#ADBACB',size=10):
        p.setPen(QColor(color)); p.setFont(QFont('Microsoft YaHei UI',size)); p.drawText(QPointF(x,y),s)
    def line(self,p,a,b,color,width=1,style=Qt.SolidLine):
        p.setPen(QPen(QColor(color),width,style)); p.drawLine(QPointF(*a),QPointF(*b))
    def curve(self,p,x,y,rect,color,lo,hi,mask=None):
        path=QPainterPath(); active=False
        for i,(xx,yy) in enumerate(zip(x,y)):
            if mask is not None and not mask[i]: active=False; continue
            pt=QPointF(rect.left()+rect.width()*(xx-x[0])/(x[-1]-x[0]),rect.bottom()-rect.height()*(yy-lo)/(hi-lo))
            if active:path.lineTo(pt)
            else:path.moveTo(pt); active=True
        p.setPen(QPen(QColor(color),2));p.drawPath(path)
    def paintEvent(self,event):
        p=QPainter(self);p.setRenderHint(QPainter.Antialiasing)
        if self.kind in ('image','a','b'):self.paint_image(p)
        else:self.paint_chart(p)
        p.end()
    def paint_image(self,p):
        o=self.owner; w,h=self.width(),self.height();p.fillRect(self.rect(),QColor('#252930'))
        thumb=self.kind!='image';local=o.mode!=0 and not thumb
        if not local:
            scale=min(w/800,h/600)*(.90 if not thumb else 1)*self.zoom
            cx=w/2+self.pan.x();cy=h/2+self.pan.y();rr=195*scale
            p.drawImage(QRectF(cx-400*scale,cy-300*scale,800*scale,600*scale),self.image)
            if thumb:
                self.text(p,14,24,'A  /  Reference' if self.kind=='a' else 'B  /  Test',CYAN if self.kind=='a' else ORANGE,10)
                self.text(p,14,h-12,'2026-10-01   •   800 × 600   •   uint16' if self.kind=='a' else '2026-10-20   •   800 × 600   •   uint16',size=8)
                return
            self.center=(cx,cy)
            for a in range(0,360,5):
                rad=math.radians(a)
                if o.calipers.isChecked() and a%15==0:
                    self.line(p,(cx+(rr-12)*math.cos(rad),cy+(rr-12)*math.sin(rad)),(cx+(rr+12)*math.cos(rad),cy+(rr+12)*math.sin(rad)),'#687381')
                for b,col,show in [(0,CYAN,o.show_a.isChecked()),(1,ORANGE,o.show_b.isChecked())]:
                    if not show:continue
                    radius=rr+(delta[a]/.1*scale if b else 0)
                    x=cx+radius*math.cos(rad);y=cy+radius*math.sin(rad)
                    if not valid[a] and b:
                        if o.reject.isChecked():self.line(p,(x-3,y-3),(x+3,y+3),'#E36B74',1.5);self.line(p,(x-3,y+3),(x+3,y-3),'#E36B74',1.5)
                        continue
                    p.setPen(Qt.NoPen);p.setBrush(QColor(col));p.drawEllipse(QPointF(x,y),2,2)
            if o.fit.isChecked():
                p.setBrush(Qt.NoBrush);p.setPen(QPen(QColor('#86A3AE'),1,Qt.DashLine));p.drawEllipse(QPointF(cx,cy),rr,rr)
            a=math.radians(o.angle);x=cx+rr*math.cos(a);y=cy+rr*math.sin(a)
            self.line(p,(cx,cy),(x,y),'#91A6BC',1,Qt.DashLine)
            p.setPen(QPen(QColor('#FFFFFF'),1.5));p.setBrush(Qt.NoBrush);p.drawEllipse(QPointF(x,y),11,11)
            self.text(p,20,27,'统一坐标 / A 原始灰度背景    ·    边缘点按真实比例绘制',size=9)
            self.text(p,20,h-18,f'选中 {o.angle:05.1f}°   |   滚轮缩放 · 右键拖动 · 单击联动',size=9)
            self.line(p,(w-130,h-30),(w-130+100*scale,h-30),'#D7DEE7',2);self.text(p,w-130,h-40,'10.0 μm',size=9)
        else:
            # Synthetic local view in tangential / outward-normal coordinates.
            cx=w/2+self.pan.x();cy=h*.48+self.pan.y();self.center=(cx,cy)
            for yy in range(0,h,3):
                intensity=int(46+99/(1+math.exp(max(-50,min(50,-(yy-cy)/7)))))
                p.fillRect(0,yy,w,3,QColor(intensity,intensity,intensity))
            shift=float(delta[int(o.angle)%360])/.1*28*self.zoom
            for i in range(-16,17):
                x=cx+i*27*self.zoom;y=cy+14*math.sin(i*.10)
                if x<15 or x>w-15:continue
                if o.calipers.isChecked():self.line(p,(x,y-35),(x,y+65),'#73808A',1)
                for b,col,show in [(0,CYAN,o.show_a.isChecked()),(1,ORANGE,o.show_b.isChecked())]:
                    if not show:continue
                    py=y+b*shift
                    if b and i in (4,5):
                        if o.reject.isChecked():self.line(p,(x-4,py-4),(x+4,py+4),'#F07580',2);self.line(p,(x-4,py+4),(x+4,py-4),'#F07580',2)
                    else:
                        p.setPen(QPen(QColor(col),1.5));p.setBrush(QColor('#252930'));p.drawEllipse(QPointF(x,py),3.6,3.6)
                if o.mode==1 and i%3==0:
                    self.line(p,(x,y),(x,y+shift),'#F1D1AD',1.3)
            self.line(p,(cx,cy-55),(cx,cy+110),'#FFFFFF',1,Qt.DashLine)
            p.setBrush(Qt.NoBrush);p.setPen(QPen(QColor('#FFFFFF'),1));p.drawEllipse(QPointF(cx,cy+shift),10,10)
            self.text(p,20,27,f'局部切向视图 / {o.angle:.1f}°    ·    原始点位真实比例',size=10)
            self.text(p,20,53,'青色 A 有效点    橙色 B 有效点    红色 × 剔除点',size=9)
            self.text(p,20,h-46,f'Δn = {delta[int(o.angle)%360]:+.4f} μm  /  {delta[int(o.angle)%360]/.1:+.4f} px','#FFFFFF',12)
            self.text(p,20,h-22,'法向：向孔外为正  |  图像 Y 向下  |  1 px = 28 屏幕像素（初始）',size=9)
            if o.mode==1:
                self.line(p,(w-190,90),(w-190,90+shift*4),'#EED4B3',2)
                self.text(p,w-315,77,'位移示意 ×4（仅右侧线段）',size=9)
            else:self.text(p,w-290,78,'白虚线：A/B 对应物理采样轴',size=9)
    def paint_chart(self,p):
        o=self.owner;w,h=self.width(),self.height();p.fillRect(self.rect(),QColor('#FFFFFF'))
        if o.mode<2:
            r=QRectF(66,36,w-96,h-75);self.chartrect=r
            low,high=(-.12,.18) if o.curve.currentIndex()==0 else (0,.18)
            if o.curve.currentIndex()==2:low,high=19.35,19.7
            for value in np.linspace(low,high,5):
                y=r.bottom()-(value-low)/(high-low)*r.height();self.line(p,(r.left(),y),(r.right(),y),'#E5EAF0')
                self.text(p,8,y+4,f'{value:.3f}','#627084',8)
            start,end=o.angle_range
            mask=(angles>=start)&(angles<=end);xx=angles[mask]
            yy=delta if o.curve.currentIndex()==0 else np.abs(delta)
            if o.curve.currentIndex()==2:
                self.curve(p,xx,np.full(len(xx),19.5),r,CYAN,low,high,valid[mask]);yy=19.5+delta
            self.curve(p,xx,yy[mask],r,ORANGE,low,high,valid[mask])
            if low<=0<=high:
                y=r.bottom()-(0-low)/(high-low)*r.height();self.line(p,(r.left(),y),(r.right(),y),'#8291A3',1,Qt.DashLine)
            if start<=o.angle<=end:
                x=r.left()+(o.angle-start)/(end-start)*r.width();self.line(p,(x,r.top()),(x,r.bottom()),'#087EF4',1,Qt.DashLine)
            for a in np.linspace(start,end,7):self.text(p,r.left()+(a-start)/(end-start)*r.width()-9,h-17,f'{a:.0f}°','#627084',8)
            self.text(p,12,20,'边缘位移 μm' if o.curve.currentIndex()<2 else '统一参考径向位置 μm','#283544',10)
            self.text(p,w-350,20,'250°–270° 缺失区不连接 · 点击 / 悬停联动','#687587',9)
        else:
            d=delta[int(o.angle)%360];xx=np.linspace(-1,1.5,301)
            a=8000+42000/(1+np.exp(-xx/.09))
            b=8000+34000/(1+np.exp(-(xx-d)/.13))+6500/(1+np.exp(-(xx-.8)/.045))
            for k in range(2):
                r=QRectF(66+k*w/2,38,w/2-100,h-82)
                ya,yb=(a,b) if k==0 else (np.gradient(a,xx),np.gradient(b,xx))
                hi=65535 if k==0 else 125000
                for t in (0,.5,1):
                    y=r.bottom()-t*r.height();self.line(p,(r.left(),y),(r.right(),y),'#E5EAF0');self.text(p,r.left()-56,y+4,f'{t*hi:,.0f}','#627084',8)
                self.curve(p,xx,ya,r,CYAN,0,hi);self.curve(p,xx,yb,r,ORANGE,0,hi)
                for pos,col in [(0,CYAN),(d,ORANGE)]:
                    x=r.left()+(pos+1)/2.5*r.width();self.line(p,(x,r.top()),(x,r.bottom()),col,1,Qt.DashLine)
                self.text(p,r.left(),22,'灰度 / 原始 DN' if k==0 else '梯度 / DN·μm⁻¹','#283544',10)
                for pos in (-1,0,1):self.text(p,r.left()+(pos+1)/2.5*r.width()-10,h-20,f'{pos:+.1f}','#627084',8)
                self.text(p,r.right()-110,h-20,'法向距离 μm','#627084',8)
                if k==1:self.text(p,r.right()-155,58,'次峰 ≈ +0.8 μm','#9D691F',9)
    def wheelEvent(self,e):
        if self.kind=='image':self.zoom=max(.5,min(8,self.zoom*(1.2 if e.angleDelta().y()>0 else 1/1.2)))
        elif self.owner.mode<2:
            a=self.owner.angle;half=45 if e.angleDelta().y()>0 else 180;self.owner.angle_range=(max(0,a-half),min(359,a+half))
        self.update()
    def mousePressEvent(self,e):
        if e.button()==Qt.RightButton:self.last=e.position();return
        self.select(e.position())
    def mouseMoveEvent(self,e):
        if self.last is not None:
            self.pan+=e.position()-self.last;self.last=e.position();self.update()
        elif self.kind=='chart' and self.owner.mode<2:self.select(e.position())
    def mouseReleaseEvent(self,e):self.last=None
    def select(self,pt):
        if self.kind=='chart' and self.owner.mode<2 and hasattr(self,'chartrect'):
            r=self.chartrect;start,end=self.owner.angle_range
            self.owner.angle=float(np.clip(start+(pt.x()-r.left())/r.width()*(end-start),0,359))
        elif self.kind=='image' and self.owner.mode==0:
            cx,cy=self.center;self.owner.angle=math.degrees(math.atan2(pt.y()-cy,pt.x()-cx))%360
        else:return
        self.owner.refresh()

class Window(QWidget):
    def __init__(self):
        super().__init__();self.mode=0;self.angle=135.;self.angle_range=(0,359)
        self.setWindowTitle('SOMA Diagnostics · DEMO / 模拟数据');self.resize(1500,1020)
        self.setStyleSheet('QWidget {font-family:"Microsoft YaHei UI";font-size:10pt;color:#20242B;background:#F5F7FA;} QPushButton,QComboBox {background:white;border:1px solid #D6DCE4;border-radius:6px;padding:7px 12px;} QPushButton:checked{background:#EAF3FF;color:#087EF4;border-color:#087EF4;} QCheckBox{spacing:5px;}')
        root=QVBoxLayout(self);root.setContentsMargins(22,14,22,12);root.setSpacing(10)
        title=QHBoxLayout();brand=label('SOMA  /  Vision Metrology',15,'#20242B',True);brand.setMinimumWidth(470);title.addWidget(brand);title.addStretch();session=label('工程模式    /    独立 A/B 会话',10);session.setMinimumWidth(230);title.addWidget(session);demo=label('DEMO / 模拟数据',10,'#996213',True);demo.setMinimumWidth(160);title.addWidget(demo);root.addLayout(title)
        row=QHBoxLayout();cn=label('成像诊断',19,'#20242B',True);cn.setMinimumWidth(130);row.addWidget(cn);en=label('SOMA Diagnostics',12);en.setMinimumWidth(240);row.addWidget(en);row.addStretch();hint=label('原始灰度保留 · 刚性坐标映射 · 亚像素边缘对比',10);hint.setMinimumWidth(440);row.addWidget(hint);root.addLayout(row)
        tools=QHBoxLayout()
        for t in ('导入 Image A','导入 Image B','B 方向：90°','配准：独立基准点','开始分析','导出结果'):
            b=QPushButton(t);tools.addWidget(b)
            if t=='开始分析':b.setStyleSheet('background:#087EF4;color:white;border-color:#087EF4;font-weight:600;')
            b.clicked.connect(lambda checked=False,t=t:self.notice.setText('DEMO：'+t+' 为流程占位；当前展示固定模拟会话。'))
        tools.addStretch();root.addLayout(tools)
        root.addWidget(label('共享参数副本  /  Caliper Circle    σ 1.0 px    阈值 5 DN/px    步长 0.25 px    极性 Dark → Bright    X/Y 0.1000 μm/px',9))
        body=QHBoxLayout();left=QVBoxLayout();body.addLayout(left,1)
        thumbs=QHBoxLayout();self.ta=Plot(self,'a');self.tb=Plot(self,'b');self.ta.setFixedHeight(150);self.tb.setFixedHeight(150);thumbs.addWidget(self.ta);thumbs.addWidget(self.tb);left.addLayout(thumbs)
        tabs=QHBoxLayout();self.group=QButtonGroup(self)
        for i,t in enumerate(('Edge Overlay / 边缘叠加','Edge Displacement / 局部位移','Edge Profile / 灰度与梯度')):
            b=QPushButton(t);b.setCheckable(True);self.group.addButton(b,i);tabs.addWidget(b)
        self.group.button(0).setChecked(True);self.group.idClicked.connect(self.set_mode);left.addLayout(tabs)
        options=QHBoxLayout()
        for name,text,checked in [('show_a','A 边缘',True),('show_b','B 边缘',True),('fit','拟合圆',True),('reject','剔除点',True),('calipers','卡尺',False)]:
            c=QCheckBox(text);c.setChecked(checked);setattr(self,name,c);options.addWidget(c);c.toggled.connect(self.refresh)
        options.addStretch()
        for t in ('适应窗口','1:1','局部放大'):
            b=QPushButton(t);options.addWidget(b);b.clicked.connect(lambda checked=False,t=t:self.view_action(t))
        left.addLayout(options)
        self.canvas=Plot(self);left.addWidget(self.canvas,1)
        chartbar=QHBoxLayout();self.curve=QComboBox();self.curve.addItems(['带符号位移 B − A','绝对位移 |B − A|','A/B 统一参考径向位置']);self.curve.currentIndexChanged.connect(self.refresh);chartbar.addWidget(self.curve);chartbar.addStretch();chartbar.addWidget(label('滚轮放大角度范围 · 悬停读值 · 点击定位',9));left.addLayout(chartbar)
        self.chart=Plot(self,'chart');self.chart.setFixedHeight(230);left.addWidget(self.chart)
        side=QFrame();side.setFixedWidth(290);side.setStyleSheet('QFrame{background:white;border:1px solid #E0E4E9;border-radius:7px;} QLabel{border:none;background:transparent;}');s=QVBoxLayout(side);s.setContentsMargins(18,18,18,18);s.setSpacing(12)
        s.addWidget(label('诊断摘要',13,'#20242B',True));s.addWidget(label('●  已配准 / 模拟独立基准点',10,'#248A3D',True))
        s.addWidget(label('B → A：先无损旋转 90°\n再刚性配准：−2.40 / +1.20 px\n微调旋转 +0.08°\n3 对独立基准点 · 残差 0.08 px',9))
        for title,num in [('边缘位移 RMS',f'{np.sqrt(np.mean(values**2)):.4f} μm'),('边缘位移 P-V',f'{np.ptp(values):.4f} μm'),('最大局部偏移',f'{np.max(abs(values)):.4f} μm'),('有效对应覆盖率',f'{100*valid.mean():.2f}%')]:
            s.addWidget(label(title,9));s.addWidget(label(num,21,'#20242B',True))
        s.addWidget(label(f'Mean {values.mean():+.4f} μm   /   P95 {np.percentile(abs(values),95):.4f} μm',9))
        self.selected=label('',10,'#087EF4',True);s.addWidget(self.selected)
        self.details=label('',9);s.addWidget(self.details)
        s.addStretch();s.addWidget(label('分析口径：独立基准配准后的几何差异。圆心对齐形状差异为后续可选口径。',9))
        s.addWidget(label('单圆无法可靠求旋转；圆心对齐会消除整体平移。数值精度不代表测量不确定度。',9,'#996213'))
        body.addWidget(side);root.addLayout(body,1)
        self.notice=label('DEMO / 模拟数据  ·  图像及曲线为程序生成  ·  未调用正式检测或配准算法  ·  原图与量测状态保持独立',9);root.addWidget(self.notice);self.refresh()
    def view_action(self,t):
        if t=='局部放大':self.set_mode(1)
        else:self.canvas.zoom=1 if t=='适应窗口' else 1/max(min(self.canvas.width()/800,self.canvas.height()/600)*.9,.01);self.canvas.pan=QPointF();self.refresh()
    def set_mode(self,i):
        self.mode=i;self.group.button(i).setChecked(True);self.canvas.zoom=1;self.canvas.pan=QPointF();self.refresh()
    def refresh(self,*args):
        i=int(self.angle)%360;ok=valid[i]
        self.selected.setText(f'{self.angle:.1f}°   /   Δn {delta[i]:+.4f} μm' if ok else f'{self.angle:.1f}°  /  无有效对应')
        if self.mode==2:
            self.details.setText(f'采样：外法向 · 双线性插值\n切向平均宽 8 px · 步长 0.25 px\nA 主峰 0.0000 μm\nB 主峰 {delta[i]:+.4f} μm\nB 次峰 ≈ +0.8 μm\n10–90% 宽度：不报告\n原因：B 存在多重跃迁')
        else:self.details.setText(f'圆参数示意（DEMO）\n圆心 / 统一 A 坐标 px\nA  (400.000, 300.000)\nB  (400.000, 300.000)\nΔC  (0.000, 0.000) px\n径向均值直径 A 39.0000 μm\nB {39+2*values.mean():.4f} μm / ΔD {2*values.mean():+.4f} μm')
        self.curve.setEnabled(self.mode<2);self.canvas.update();self.chart.update()

if __name__=='__main__':
    app=QApplication(sys.argv)
    for font in ('C:/Windows/Fonts/msyh.ttc','C:/Windows/Fonts/msyhbd.ttc','C:/Windows/Fonts/segoeui.ttf'):
        if Path(font).exists():QFontDatabase.addApplicationFont(font)
    app.setFont(QFont('Microsoft YaHei',10))
    w=Window();w.show();app.processEvents()
    if '--capture' in sys.argv:
        out=Path(__file__).parent
        for i,name in enumerate(['01-overview.png','02-local-displacement.png','03-profile-gradient.png']):
            w.set_mode(i);app.processEvents()
            if not w.grab().save(str(out/name)):raise RuntimeError(name)
        print(out);w.close()
    else:sys.exit(app.exec())
