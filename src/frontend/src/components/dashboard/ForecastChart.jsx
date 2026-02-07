import React from 'react';
import { Card, CardHeader } from '../ui/Card';
import {
    AreaChart,
    Area,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer
} from 'recharts';

export const ForecastChart = ({ data }) => {
    return (
        <Card className="col-span-1 lg:col-span-2 min-h-[400px]">
            <CardHeader
                title="Demand Forecast"
                subtitle="AI-powered prediction for next 7 days"
                action={
                    <div className="flex items-center space-x-2 text-xs">
                        <span className="flex items-center"><span className="w-2 h-2 rounded-full bg-eagle-green mr-1"></span> Prediction</span>
                        <span className="flex items-center"><span className="w-2 h-2 rounded-full bg-emerald-900 mr-1"></span> Confidence</span>
                    </div>
                }
            />

            <div className="h-[300px] w-full mt-4">
                <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={data}>
                        <defs>
                            <linearGradient id="colorDemand" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor="#00E054" stopOpacity={0.3} />
                                <stop offset="95%" stopColor="#00E054" stopOpacity={0} />
                            </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" stroke="#333" vertical={false} />
                        <XAxis
                            dataKey="day_name"
                            tick={{ fill: '#9ca3af', fontSize: 12 }}
                            axisLine={false}
                            tickLine={false}
                            dy={10}
                            tickFormatter={(val) => val.slice(0, 3)}
                        />
                        <YAxis
                            tick={{ fill: '#9ca3af', fontSize: 12 }}
                            axisLine={false}
                            tickLine={false}
                            dx={-10}
                        />
                        <Tooltip
                            contentStyle={{
                                backgroundColor: 'rgba(24, 24, 27, 0.9)',
                                borderColor: 'rgba(255,255,255,0.1)',
                                backdropFilter: 'blur(8px)',
                                borderRadius: '8px',
                                color: '#fff'
                            }}
                            itemStyle={{ color: '#00E054' }}
                        />
                        <Area
                            type="monotone"
                            dataKey="upper_bound"
                            stroke="transparent"
                            fill="#064e3b"
                            fillOpacity={0.5}
                        />
                        <Area
                            type="monotone"
                            dataKey="lower_bound"
                            stroke="transparent"
                            fill="transparent"
                        />
                        <Area
                            type="monotone"
                            dataKey="predicted_demand"
                            stroke="#00E054"
                            strokeWidth={3}
                            fillOpacity={1}
                            fill="url(#colorDemand)"
                        />
                    </AreaChart>
                </ResponsiveContainer>
            </div>
        </Card>
    );
};
